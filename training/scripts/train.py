"""Train the vision models that power the automatic analysis.

Three models, each a separate run:

    players : YOLO detection over ball / goalkeeper / player / referee.
              Trained at a high input size because the ball is a handful of
              pixels wide in a broadcast frame.
    pitch   : YOLO pose over the 32 pitch landmarks. Its keypoints are what
              turn image coordinates into metres on the pitch.
    ball    : a specialist single-class detector for the one object everything
              depends on. Trained on native-resolution 640 windows rather than
              downscaled frames — see prepare_ball_crops.py for why that is the
              whole ballgame — so run that script before training this.

Usage
    python training/scripts/train.py players --epochs 120 --imgsz 960
    python training/scripts/train.py players --resume
    python training/scripts/train.py all

A note on resuming. The first few epochs after `--resume` look like a collapse:
validation loss spikes and mAP falls close to zero before climbing back over
roughly ten epochs. Training loss stays where it was, which is the giveaway —
what resets is the exponential moving average of the weights that validation is
run against, not the model. `best.pt` is kept by fitness, so those epochs cannot
overwrite a better earlier one, and the run recovers on its own. Nothing needs
doing about it beyond not panicking.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

# Must be set before torch is imported. Apple's MPS allocator keeps freed
# blocks in a cache and, left to itself, lets that cache grow until the system
# kills the process — which is exactly how both models died here, at 16.9 GB on
# a 16 GB machine. These two ratios are the documented brake: once allocations
# pass the high watermark PyTorch releases cached blocks until it is back under
# the low one, instead of asking the OS for memory that does not exist.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.75")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.55")
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"
RUNS = ROOT / "training" / "runs"
WEIGHTS = ROOT / "models"

# Small datasets (250-1000 images) with a fixed camera style: strong colour and
# geometric augmentation matters far more than a bigger backbone.
RECIPES = {
    "players": dict(
        base="yolo11s.pt",
        data=DATASETS / "players" / "data.yaml",
        epochs=100,
        # 640 rather than the 960 the ball would prefer: measured on the target
        # machine (Apple M2, 16 GB unified memory), 960 x batch 8 needs ~10 GB
        # and drives the system into swap, where an epoch takes hours instead of
        # two minutes. The specialist ball model recovers the small-object
        # accuracy that the lower resolution costs.
        imgsz=640,
        batch=8,
        task="detect",
        # Broadcast footage is always upright and never mirrored mid-clip, but
        # left/right flips double an already tiny dataset for free.
        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,
        scale=0.5,
        mosaic=1.0,
        close_mosaic=15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    ),
    "pitch": dict(
        base="yolo11s-pose.pt",
        data=DATASETS / "pitch" / "data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        task="pose",
        # Pose training is forced onto the CPU. Ultralytics has a long-standing
        # bug on Apple's MPS backend (ultralytics#4031) where the backward pass
        # of a keypoint model raises "view size is not compatible with input
        # tensor's size and stride"; the library itself warns and recommends
        # device=cpu. Measured on an M2, a CPU epoch here takes about 170 s
        # against roughly 95 s on MPS, so the cost is real but small — and a
        # model that trains beats one that crashes. Override with --device if a
        # CUDA machine is available, where MPS is irrelevant.
        device="cpu",
        # Mosaic splices four images together, which invents pitch geometry that
        # cannot exist. For landmark regression that is actively harmful.
        mosaic=0.0,
        fliplr=0.5,
        flipud=0.0,
        degrees=3.0,
        scale=0.4,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    ),
    "ball": dict(
        base="yolo11n.pt",
        # Native-resolution windows, not whole frames. Training on the frames
        # shrinks a twelve-pixel ball to four and leaves the model useless at
        # the size it is actually asked about; prepare_ball_crops.py explains
        # the measurement that established this.
        data=DATASETS / "ball_crops" / "data.yaml",
        # More epochs than the frame version needed: there are six times as
        # many samples now, and a third of them are negatives the model has
        # never had to learn to reject before.
        epochs=80,
        imgsz=640,
        batch=16,
        task="detect",
        # Full precision. Ultralytics turns automatic mixed precision on by
        # default, and on Apple's MPS backend this run diverged to NaN losses
        # at epoch 9 — every loss term at once, from a healthy epoch 8, which
        # is the signature of a numerical blow-up rather than a bad batch. The
        # eight epochs before it were already at 0.95 mAP50, so the divergence
        # cost nothing except the hours that would have followed it; leaving
        # amp on would have spent those hours producing NaN.
        amp=False,
        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,
        # No scale jitter worth the name. Every other recipe wants the model
        # robust to object size; this one wants the opposite. The ball is
        # always about twelve pixels in a native-resolution window, that is the
        # single fact the model most needs to learn, and augmenting it away is
        # how the first version ended up guessing.
        scale=0.15,
        # Mosaic tiles four images into one and rescales them, which does the
        # same damage as scale jitter and additionally puts four balls in a
        # frame that will only ever contain zero or one.
        mosaic=0.0,
        close_mosaic=0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    ),
}


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_one(name: str, overrides: dict) -> Path:
    recipe = dict(RECIPES[name])
    recipe.pop("task")
    requested_base = overrides.get("base")
    # A caller-supplied starting point wins over the recipe's. It is read
    # rather than popped, so `all` gives every model the same treatment, and
    # excluded from the overrides folded in below so it never reaches
    # model.train() as a stray keyword.
    overrides = {k: v for k, v in overrides.items() if k != "base"}
    base = str(requested_base or RECIPES[name]["base"])
    recipe.pop("base", None)
    # A recipe may pin a device (see the pitch model). An explicit --device on
    # the command line still overrides it; the auto-detected default does not.
    pinned_device = recipe.get("device")
    recipe.update({k: v for k, v in overrides.items() if v is not None})
    if pinned_device and not overrides.get("device_explicit"):
        recipe["device"] = pinned_device

    device = recipe.pop("device")
    recipe.pop("device_explicit", None)
    resume = bool(recipe.pop("resume", False))
    run_name = recipe.pop("run_name", None) or f"{name}"

    # Resuming loads the checkpoint's own optimiser state and epoch counter, so
    # every hyper-parameter comes from the interrupted run rather than from the
    # recipe. Long trainings on a laptop get interrupted; this makes that cheap.
    checkpoint = RUNS / run_name / "weights" / "last.pt"
    if resume:
        if not checkpoint.exists():
            raise FileNotFoundError(f"nothing to resume: {checkpoint} does not exist")
        base = str(checkpoint)
        recipe = {"resume": True}

    print(f"\n{'=' * 70}\n  {'resuming' if resume else 'training'} '{name}'  ({base}, device={device})\n{'=' * 70}")
    started = time.time()

    model = YOLO(base)

    # Apple's MPS allocator caches aggressively and does not give it back on
    # its own. Left alone, this run climbed from 4.3 GB at epoch 1 to 16.9 GB
    # at epoch 21 on a 16 GB machine and was killed by the system — twice, once
    # per model, each time discarding hours of work that had already produced a
    # better checkpoint than the one deployed.
    #
    # Emptying the cache between epochs costs a fraction of a second and holds
    # the footprint flat. gc first, because the allocator can only release
    # blocks that nothing still references.
    if device == "mps":
        released = {"n": 0}

        def _release(_trainer) -> None:
            gc.collect()
            torch.mps.empty_cache()
            released["n"] += 1
            if released["n"] <= 2:
                # Say it once or twice, so a silent no-op is visible in the log
                # rather than mistaken for a fix that worked.
                print(f"[mem] released MPS cache "
                      f"({torch.mps.driver_allocated_memory() / 1e9:.2f} GB held)")

        model.add_callback("on_train_epoch_end", _release)
        model.add_callback("on_val_end", _release)

    model.train(
        project=str(RUNS),
        name=run_name,
        exist_ok=True,
        device=device,
        # Zero on MPS. Worker processes each hold their own copy of the batch
        # being assembled, which is memory this machine does not have to spare,
        # and the decode is not the bottleneck here anyway — the crops are
        # small. They were also the source of the leaked-semaphore warning that
        # accompanied both out-of-memory kills.
        workers=0 if device == "mps" else 2,
        cache=False,
        patience=40,
        seed=0,
        plots=True,
        val=True,
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in recipe.items()},
    )

    best = RUNS / run_name / "weights" / "best.pt"
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    target = WEIGHTS / f"{name}.pt"
    if best.exists():
        target.write_bytes(best.read_bytes())
        print(f"\n[ok] {name}: {best}  ->  {target}")
    mins = (time.time() - started) / 60
    print(f"[ok] {name} finished in {mins:.1f} min")
    return target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=[*RECIPES, "all"])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--device")
    ap.add_argument("--run-name", dest="run_name")
    ap.add_argument(
        "--base",
        help="start from these weights instead of the recipe's. Unlike --resume "
             "this keeps the recipe's hyper-parameters, which is what you want "
             "after a run died on memory and has to come back with a smaller "
             "batch than the one it was killed holding.",
    )
    ap.add_argument(
        "--resume", action="store_true",
        help="continue an interrupted run from its last checkpoint",
    )
    args = ap.parse_args()

    overrides = dict(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=pick_device(args.device),
        device_explicit=bool(args.device),
        run_name=args.run_name,
        resume=args.resume,
        base=args.base,
    )

    names = list(RECIPES) if args.model == "all" else [args.model]
    for name in names:
        train_one(name, overrides)


if __name__ == "__main__":
    main()
