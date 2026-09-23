"""The ball detector, measured in the regimes it is actually used in.

`evaluate.py` reports mAP over whole test images, which is the number every
detector reports and, for this model, the wrong one. The ball model is almost
never shown a whole image. `Detector` cuts a 640-square window at native
resolution around the ball's last known position and asks about that; only when
the trail goes cold does it sweep, and it sweeps as a grid of the same windows.
A figure computed on downscaled full frames says nothing about either.

So this measures the model the way it is called, and separates the two failure
modes that a single recall number hides:

    found the ball   a box overlapping the true one by at least half
    wrong box        a confident detection somewhere else entirely
    found nothing    no detection above the threshold

The distinction matters more than the totals. Finding nothing costs a frame of
possession, which the ball interpolation can bridge. A wrong box is worse than
silence: it puts the ball on a player who never touched it, and possession,
passes and every event derived from them follow it there.

This is the measurement that found the original bug. The first ball model was
trained on full 1920x1080 frames at imgsz=640 — a four-pixel ball — and then
asked about native-resolution windows holding a twelve-pixel one. It scored
0.382 mAP, which looked poor but survivable, and 60% wrong boxes in the regime
that handles nearly every frame, which is not survivable at all.

    python training/scripts/evaluate_ball.py
    python training/scripts/evaluate_ball.py --weights training/runs/ball/weights/best.pt
    python training/scripts/evaluate_ball.py --compare models/ball.pt other.pt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import supervision as sv
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from football_ai.detect import DetectorConfig, tile_windows   # noqa: E402

DATASET = ROOT / "datasets" / "ball" / "test"
WINDOW = 640


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (
        (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    )
    return float(inter / union) if union > 0 else 0.0


def best_box(result) -> tuple[np.ndarray | None, float]:
    det = sv.Detections.from_ultralytics(result)
    if len(det) == 0:
        return None, 0.0
    i = int(np.argmax(det.confidence))
    return det.xyxy[i].astype(np.float32), float(det.confidence[i])


def load(dataset: Path, limit: int | None = None):
    """Every test frame with a labelled ball, as `(image_bgr, box_px)`."""
    items = []
    for label_path in sorted((dataset / "labels").glob("*.txt")):
        image_path = next(
            (p for p in (dataset / "images" / f"{label_path.stem}{e}"
                         for e in (".jpg", ".jpeg", ".png")) if p.exists()),
            None,
        )
        if image_path is None:
            continue
        lines = [line for line in label_path.read_text().splitlines() if line.strip()]
        if not lines:
            continue
        _, cx, cy, w, h = (float(v) for v in lines[0].split()[:5])
        image = np.array(Image.open(image_path).convert("RGB"))[:, :, ::-1]
        H, W = image.shape[:2]
        items.append((
            np.ascontiguousarray(image),
            np.array([(cx - w / 2) * W, (cy - h / 2) * H,
                      (cx + w / 2) * W, (cy + h / 2) * H], dtype=np.float32),
        ))
        if limit and len(items) >= limit:
            break
    return items


class Regimes:
    """Each way the detector actually calls the model."""

    def __init__(self, model: YOLO, conf: float, jitter: int, seed: int = 0):
        self.model = model
        self.conf = conf
        self.jitter = jitter
        self.rng = random.Random(seed)

    def _predict(self, images, imgsz):
        return self.model.predict(images, imgsz=imgsz, conf=self.conf, verbose=False)

    def full_frame(self, image, _truth):
        """What `evaluate.py` measures, and what training used to optimise."""
        return best_box(self._predict(image, 640)[0])

    def window(self, image, truth):
        """The hot path: one native-resolution window near the last sighting.

        The window is deliberately not centred on the ball. At inference it is
        aimed at where the ball *was*, and the ball has moved since — centring
        it here would measure a problem the detector never actually faces.
        """
        H, W = image.shape[:2]
        cx = (truth[0] + truth[2]) / 2 + self.rng.uniform(-self.jitter, self.jitter)
        cy = (truth[1] + truth[3]) / 2 + self.rng.uniform(-self.jitter, self.jitter)
        x0 = int(np.clip(cx - WINDOW / 2, 0, max(W - WINDOW, 0)))
        y0 = int(np.clip(cy - WINDOW / 2, 0, max(H - WINDOW, 0)))
        box, conf = best_box(self._predict(image[y0:y0 + WINDOW, x0:x0 + WINDOW], WINDOW)[0])
        if box is None:
            return None, 0.0
        return box + np.array([x0, y0, x0, y0], np.float32), conf

    def sweep(self, image, _truth):
        """Reacquisition: a grid of windows, no help from a previous position."""
        H, W = image.shape[:2]
        tiles = tile_windows(W, H, WINDOW, DetectorConfig().ball_tile_overlap)
        results = self._predict([image[y0:y1, x0:x1] for x0, y0, x1, y1 in tiles], WINDOW)
        found, best_conf = None, 0.0
        for (x0, y0, _, _), result in zip(tiles, results):
            box, conf = best_box(result)
            if box is not None and conf > best_conf:
                found = box + np.array([x0, y0, x0, y0], np.float32)
                best_conf = conf
        return found, best_conf


def measure(regime, items, threshold: float = 0.5) -> dict:
    hit = wrong = missed = 0
    ious, confidences = [], []
    for image, truth in items:
        box, conf = regime(image, truth)
        if box is None:
            missed += 1
            continue
        overlap = iou(box, truth)
        ious.append(overlap)
        confidences.append(conf)
        if overlap >= threshold:
            hit += 1
        else:
            wrong += 1
    n = max(len(items), 1)
    return {
        "found": hit / n, "wrong": wrong / n, "missed": missed / n,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_conf": float(np.mean(confidences)) if confidences else 0.0,
        "n": len(items),
    }


def report(name: str, weights: Path, items, conf: float, jitter: int) -> dict:
    model = YOLO(str(weights))
    regimes = Regimes(model, conf, jitter)
    out = {}
    print(f"\n  {name}  ({weights})")
    print(f"  {'regime':<28} {'found':>7} {'wrong box':>10} {'nothing':>9} "
          f"{'mean IoU':>9} {'conf':>6}")
    for label, fn in (
        ("full frame @ 640", regimes.full_frame),
        ("native window (hot path)", regimes.window),
        ("tiled sweep (reacquire)", regimes.sweep),
    ):
        r = measure(fn, items)
        out[label] = r
        print(f"  {label:<28} {r['found']:>6.0%} {r['wrong']:>10.0%} "
              f"{r['missed']:>9.0%} {r['mean_iou']:>9.2f} {r['mean_conf']:>6.2f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=ROOT / "models" / "ball.pt")
    ap.add_argument("--compare", type=Path, nargs="*", default=None,
                    help="two or more weight files to measure side by side")
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--conf", type=float, default=DetectorConfig().ball_conf)
    ap.add_argument("--jitter", type=int, default=120,
                    help="how far the window centre sits from the ball, in pixels")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--json", type=Path, dest="json_out")
    args = ap.parse_args()

    if not args.dataset.is_dir():
        print(f"No test split at {args.dataset}", file=sys.stderr)
        return 1

    items = load(args.dataset, args.limit)
    if not items:
        print("No labelled test frames found.", file=sys.stderr)
        return 1
    print(f"\n  {len(items)} test frames from {args.dataset}")

    targets = args.compare or [args.weights]
    results = {}
    for path in targets:
        if not Path(path).exists():
            print(f"  (skipping missing {path})")
            continue
        results[str(path)] = report(Path(path).stem, Path(path), items, args.conf, args.jitter)

    print("\n  A wrong box is worse than nothing: the interpolation can bridge a")
    print("  frame with no ball, but it cannot know that a confident detection")
    print("  on somebody's sock is not the ball.\n")

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"  wrote {args.json_out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
