"""Rebuild the ball dataset as native-resolution windows, matching inference.

The ball dataset ships full 1920x1080 broadcast frames in which the ball is
about twelve pixels across. Training on those at imgsz=640 shrinks the frame by
three, so the model only ever learns a **four pixel** ball — barely above the
stride of the finest detection head it has.

Inference then does something completely different. `Detector._ball_from_crop`,
which handles almost every frame, cuts a 640x640 window at *native* resolution
around the ball's last known position and runs the model on that. The ball in
that window is its true twelve pixels — three times the size of anything the
model was trained on. Measured on the held-out test set, that mismatch costs
almost everything:

    full frame @ 640    recall 26%   wrong box  4%
    full frame @ 1280   recall 62%   wrong box 22%
    native 640 crop     recall 34%   wrong box 60%   <- the path actually used

Sixty percent wrong. The model finds *something* in nearly every window and it
is usually not the ball, which is worse than finding nothing: possession,
passes and every event derived from them are being placed off a phantom.

This script removes the mismatch by training the model on exactly what it will
be asked about — 640x640 windows at native resolution. That is also cheaper
than raising imgsz to 1920 would be, for the same effective detail.

Two details matter as much as the resolution:

**The ball is not centred.** At inference the window is centred on where the
ball *was*, and it has moved since. Centring it in every training sample would
teach a centre prior that the real windows violate constantly, so each crop
places the ball at a random offset.

**Negatives are included.** Every one of the 989 source images contains a ball,
so the model has never once been shown a football pitch and told there is no
ball in it — and at inference it meets that case constantly, whenever the ball
is occluded or has left the window. A detector that has never seen a negative
answers every question with a ball. Windows drawn from elsewhere in the same
frames fix that, and they are drawn from the same frames on purpose: the false
positives worth learning to reject are the players' socks, the line markings
and the bright patches in the crowd, not some unrelated photograph.

    python training/scripts/prepare_ball_crops.py
    python training/scripts/prepare_ball_crops.py --crops 4 --negatives 2
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"
SOURCE = DATASETS / "ball"
DEST = DATASETS / "ball_crops"

SPLITS = ("train", "valid", "test")
WINDOW = 640


def read_label(path: Path, width: int, height: int) -> np.ndarray | None:
    """The ball box in pixels, or None if this frame has no ball labelled."""
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return None
    _, cx, cy, w, h = (float(v) for v in lines[0].split()[:5])
    return np.array([
        (cx - w / 2) * width, (cy - h / 2) * height,
        (cx + w / 2) * width, (cy + h / 2) * height,
    ], dtype=np.float32)


def window_at(centre_x: float, centre_y: float, width: int, height: int) -> tuple[int, int]:
    """Top-left of a WINDOW-sized crop centred as close to the point as the
    frame allows."""
    x = int(np.clip(centre_x - WINDOW / 2, 0, max(width - WINDOW, 0)))
    y = int(np.clip(centre_y - WINDOW / 2, 0, max(height - WINDOW, 0)))
    return x, y


def to_yolo(box: np.ndarray, x0: int, y0: int) -> str:
    x1, y1, x2, y2 = box - np.array([x0, y0, x0, y0], dtype=np.float32)
    cx, cy = (x1 + x2) / 2 / WINDOW, (y1 + y2) / 2 / WINDOW
    w, h = (x2 - x1) / WINDOW, (y2 - y1) / WINDOW
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def inside(box: np.ndarray, x0: int, y0: int, margin: int = 2) -> bool:
    """Is the whole ball inside this window, with a little room to spare?

    A ball clipped by the window edge is a half-ball, and teaching the model
    that a half-ball is a ball invites it to fire on every bright fragment.
    """
    return (
        box[0] >= x0 + margin and box[1] >= y0 + margin
        and box[2] <= x0 + WINDOW - margin and box[3] <= y0 + WINDOW - margin
    )


def crops_for(
    image: np.ndarray, box: np.ndarray, rng: random.Random,
    positives: int, negatives: int, jitter: int,
) -> list[tuple[np.ndarray, str | None]]:
    """Windows cut from one frame: `positives` containing the ball, `negatives`
    guaranteed not to."""
    height, width = image.shape[:2]
    ball_x, ball_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    out: list[tuple[np.ndarray, str | None]] = []

    attempts = 0
    while sum(1 for _, label in out if label) < positives and attempts < positives * 12:
        attempts += 1
        x0, y0 = window_at(
            ball_x + rng.uniform(-jitter, jitter),
            ball_y + rng.uniform(-jitter, jitter),
            width, height,
        )
        if not inside(box, x0, y0):
            continue
        out.append((image[y0:y0 + WINDOW, x0:x0 + WINDOW].copy(), to_yolo(box, x0, y0)))

    attempts = 0
    taken = 0
    while taken < negatives and attempts < negatives * 20:
        attempts += 1
        x0 = rng.randint(0, max(width - WINDOW, 0))
        y0 = rng.randint(0, max(height - WINDOW, 0))
        # Any overlap at all disqualifies it: a window holding part of the ball
        # labelled "empty" teaches the model to suppress the thing we want.
        if not (box[2] < x0 or box[0] > x0 + WINDOW or box[3] < y0 or box[1] > y0 + WINDOW):
            continue
        out.append((image[y0:y0 + WINDOW, x0:x0 + WINDOW].copy(), None))
        taken += 1

    return out


def build(split: str, positives: int, negatives: int, jitter: int, seed: int) -> dict:
    src_images, src_labels = SOURCE / split / "images", SOURCE / split / "labels"
    if not src_images.is_dir():
        return {}

    dst_images, dst_labels = DEST / split / "images", DEST / split / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    written = {"positive": 0, "negative": 0, "frames": 0, "skipped": 0}

    for label_path in sorted(src_labels.glob("*.txt")):
        image_path = next(
            (p for p in (src_images / f"{label_path.stem}{ext}"
                         for ext in (".jpg", ".jpeg", ".png")) if p.exists()),
            None,
        )
        if image_path is None:
            continue
        image = cv2.imread(str(image_path))
        if image is None or min(image.shape[:2]) < WINDOW:
            written["skipped"] += 1
            continue

        box = read_label(label_path, image.shape[1], image.shape[0])
        if box is None:
            written["skipped"] += 1
            continue

        written["frames"] += 1
        for i, (crop, label) in enumerate(
            crops_for(image, box, rng, positives, negatives, jitter)
        ):
            name = f"{label_path.stem}_{i:02d}"
            cv2.imwrite(str(dst_images / f"{name}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            (dst_labels / f"{name}.txt").write_text(f"{label}\n" if label else "")
            written["positive" if label else "negative"] += 1

    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=int, default=3, help="windows containing the ball, per frame")
    ap.add_argument("--negatives", type=int, default=2, help="ball-free windows per frame")
    ap.add_argument(
        "--jitter", type=int, default=200,
        help="how far the ball may sit from the window centre, in pixels. Matches "
             "how far it can move between the frame the window was aimed at and "
             "the frame it is cut from.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="rebuild from scratch")
    args = ap.parse_args()

    if not SOURCE.is_dir():
        print(f"No ball dataset at {SOURCE}. Run download_datasets.py first.", file=sys.stderr)
        return 1
    if DEST.exists():
        if not args.force:
            print(f"{DEST} already exists — pass --force to rebuild.", file=sys.stderr)
            return 1
        shutil.rmtree(DEST)

    totals = {}
    for split in SPLITS:
        counts = build(split, args.crops, args.negatives, args.jitter, args.seed)
        if counts:
            totals[split] = counts
            print(f"[ok  ] {split}: {counts['frames']} frames -> "
                  f"{counts['positive']} with ball + {counts['negative']} without"
                  + (f"  ({counts['skipped']} skipped)" if counts["skipped"] else ""))

    (DEST / "data.yaml").write_text(yaml.safe_dump({
        "path": str(DEST),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["ball"],
    }, sort_keys=False))

    total = sum(c["positive"] + c["negative"] for c in totals.values())
    print(f"\n  {total} windows at {WINDOW}x{WINDOW}, native resolution, in {DEST}")
    print(f"  train with:  python training/scripts/train.py ball\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
