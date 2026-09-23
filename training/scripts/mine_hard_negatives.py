"""Collect the things the ball detector mistakes for a ball, and teach it back.

The training set already contains negatives — windows of pitch with no ball in
them — and they did most of the work: they took the wrong-box rate on the hot
path from 60% to 4%. But those negatives were chosen at random, and a random
patch of grass is not what the model gets wrong. What it gets wrong is a sock,
a line marking, a bright patch in the crowd, a ball-shaped piece of an
advertising hoarding. Random negatives teach it that grass is not a ball, which
it already knew.

So this asks the model itself. It is run over the training footage exactly as
the pipeline runs it — a native-resolution window near the ball — and every
confident detection that is not the ball is recorded. Those are, by definition,
the mistakes it is currently disposed to make. Cropped and labelled as empty,
they are the examples with something to teach.

**The trap, and why the true ball is kept.** A window centred on a false
positive very often still contains the real ball a few hundred pixels away.
Writing that window out as "empty" would teach the model to suppress the very
thing it exists to find — turning a fix for false positives into a cause of
misses. So every crop is checked against the frame's true ball box, and if the
ball is inside, the crop is written *with* it. A hard negative is a window
where the model was wrong, not a window where the answer is nothing.

**What running it showed, and why the default split is the wrong one.** The
model's mistake rate depends entirely on whether it has seen the frame before:

    train (trained on)          0.02 phantom balls per frame
    valid (picked the weights)  0.06
    test  (never seen)          0.15

A model does not make mistakes on data it has memorised, so mining its own
training set collects almost nothing — and what little it collects is the
residue it could not fit, not the mistakes it makes in the field. Mining the
test set would fix that and destroy the only honest measurement in the project.

The conclusion is not that the technique is wrong; it is that this dataset is
spent. Hard negatives are worth mining from footage the model has never been
trained on and never will be judged on — a match video, where a false positive
can be recognised without labels by the fact that the ball does not teleport.

    python training/scripts/mine_hard_negatives.py            # look and report
    python training/scripts/mine_hard_negatives.py --write     # add to the dataset

Mined crops are written into the existing crop dataset, so retraining is the
ordinary command afterwards:

    python training/scripts/train.py ball --base models/ball.pt --epochs 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from football_ai.detect import DetectorConfig      # noqa: E402

SOURCE = ROOT / "datasets" / "ball"
CROPS = ROOT / "datasets" / "ball_crops"
WINDOW = 640


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def truth_box(label_path: Path, width: int, height: int) -> np.ndarray | None:
    lines = [l for l in label_path.read_text().splitlines() if l.strip()]
    if not lines:
        return None
    _, cx, cy, w, h = (float(v) for v in lines[0].split()[:5])
    return np.array([
        (cx - w / 2) * width, (cy - h / 2) * height,
        (cx + w / 2) * width, (cy + h / 2) * height,
    ])


def to_yolo(box: np.ndarray, x0: int, y0: int) -> str:
    x1, y1, x2, y2 = box - np.array([x0, y0, x0, y0], dtype=float)
    cx, cy = (x1 + x2) / 2 / WINDOW, (y1 + y2) / 2 / WINDOW
    return f"0 {cx:.6f} {cy:.6f} {(x2 - x1) / WINDOW:.6f} {(y2 - y1) / WINDOW:.6f}"


def inside(box: np.ndarray, x0: int, y0: int, margin: int = 2) -> bool:
    return (
        box[0] >= x0 + margin and box[1] >= y0 + margin
        and box[2] <= x0 + WINDOW - margin and box[3] <= y0 + WINDOW - margin
    )


def mine(split: str, model: YOLO, conf: float, limit: int | None) -> list[dict]:
    """Every confident detection in `split` that is not the ball."""
    images, labels = SOURCE / split / "images", SOURCE / split / "labels"
    if not images.is_dir():
        return []

    found: list[dict] = []
    paths = sorted(labels.glob("*.txt"))
    if limit:
        paths = paths[:limit]

    for n, label_path in enumerate(paths, 1):
        image_path = next(
            (p for p in (images / f"{label_path.stem}{e}"
                         for e in (".jpg", ".jpeg", ".png")) if p.exists()),
            None,
        )
        if image_path is None:
            continue
        frame = cv2.imread(str(image_path))
        if frame is None or min(frame.shape[:2]) < WINDOW:
            continue
        height, width = frame.shape[:2]
        truth = truth_box(label_path, width, height)
        if truth is None:
            continue

        # The hot path: a native window aimed near where the ball is.
        cx, cy = (truth[0] + truth[2]) / 2, (truth[1] + truth[3]) / 2
        x0 = int(np.clip(cx - WINDOW / 2, 0, max(width - WINDOW, 0)))
        y0 = int(np.clip(cy - WINDOW / 2, 0, max(height - WINDOW, 0)))
        window = np.ascontiguousarray(frame[y0:y0 + WINDOW, x0:x0 + WINDOW])

        result = model.predict(window, imgsz=WINDOW, conf=conf, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        for i in range(len(detections)):
            box = detections.xyxy[i] + np.array([x0, y0, x0, y0], dtype=np.float32)
            if iou(box, truth) >= 0.5:
                continue
            found.append({
                "frame": image_path,
                "truth": truth,
                "box": box,
                "conf": float(detections.confidence[i]),
                "size": (width, height),
            })
        if n % 100 == 0:
            print(f"    {split}: {n}/{len(paths)} frames, {len(found)} mistakes so far")
    return found


def write(mistakes: list[dict], split: str) -> tuple[int, int]:
    """Crop each mistake into the training set. Returns (written, with_ball)."""
    out_images = CROPS / split / "images"
    out_labels = CROPS / split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    written = kept_ball = 0
    for i, m in enumerate(mistakes):
        frame = cv2.imread(str(m["frame"]))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        bx = (m["box"][0] + m["box"][2]) / 2
        by = (m["box"][1] + m["box"][3]) / 2
        x0 = int(np.clip(bx - WINDOW / 2, 0, max(width - WINDOW, 0)))
        y0 = int(np.clip(by - WINDOW / 2, 0, max(height - WINDOW, 0)))
        crop = frame[y0:y0 + WINDOW, x0:x0 + WINDOW]
        if crop.shape[0] != WINDOW or crop.shape[1] != WINDOW:
            continue

        # The trap. A window aimed at a false positive frequently still holds
        # the real ball; calling it empty would train the model to suppress it.
        label = ""
        if inside(m["truth"], x0, y0):
            label = to_yolo(m["truth"], x0, y0) + "\n"
            kept_ball += 1

        name = f"hardneg_{i:04d}_{m['frame'].stem}"
        cv2.imwrite(str(out_images / f"{name}.jpg"), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out_labels / f"{name}.txt").write_text(label)
        written += 1
    return written, kept_ball


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=ROOT / "models" / "ball.pt")
    ap.add_argument("--splits", nargs="*", default=["train"],
                    help="only train by default: mining the validation split "
                         "would tune the model on the data used to judge it.")
    ap.add_argument("--conf", type=float, default=DetectorConfig().ball_conf)
    ap.add_argument("--limit", type=int, help="frames per split, for a quick look")
    ap.add_argument("--write", action="store_true",
                    help="add the crops to datasets/ball_crops")
    args = ap.parse_args()

    if not args.weights.exists():
        print(f"No weights at {args.weights}", file=sys.stderr)
        return 1
    if not CROPS.is_dir():
        print(f"No crop dataset at {CROPS} — run prepare_ball_crops.py first",
              file=sys.stderr)
        return 1

    model = YOLO(str(args.weights))
    print(f"\n  mining {args.weights.name} for its own false positives")

    total = 0
    for split in args.splits:
        mistakes = mine(split, model, args.conf, args.limit)
        frames = len(list((SOURCE / split / "labels").glob("*.txt")))
        looked = min(args.limit, frames) if args.limit else frames
        print(f"\n  {split}: {len(mistakes)} phantom balls over {looked} frames "
              f"({len(mistakes) / max(looked, 1):.2f} per frame)")
        if mistakes:
            confs = np.array([m["conf"] for m in mistakes])
            print(f"    confidence  median {np.median(confs):.2f}  max {confs.max():.2f}")
            print(f"    the confident ones are the damaging ones: a detection at "
                  f"{confs.max():.2f}\n    outranks the real ball wherever both appear.")
        if args.write and mistakes:
            written, with_ball = write(mistakes, split)
            print(f"    wrote {written} crops to {CROPS / split}")
            print(f"    of which {with_ball} still contain the real ball and keep "
                  f"its label —\n    labelling those empty would train the misses back in.")
        total += len(mistakes)

    if not args.write:
        print(f"\n  nothing written. Pass --write to add these {total} crops.\n")
    else:
        print(f"\n  retrain with:\n"
              f"    python training/scripts/train.py ball --base models/ball.pt "
              f"--epochs 12 --batch 8\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
