"""Mine the ball detector's mistakes from match footage, with no labels at all.

`mine_hard_negatives.py` asked the model for its own false positives on the
training set and recorded why that fails: a model does not make mistakes on
data it has memorised. It found 0.02 phantom balls per frame on the split it
was trained on and 0.15 on the split it had never seen. Its own conclusion was
that hard negatives are worth mining from footage the model has never been
trained on — a match video — where a false positive can be recognised without
any labelling.

This is that. The recognising principle is physical rather than annotated:

    **a ball cannot teleport.**

A detection sitting far from the frame before it, far from the frame after it,
while those two neighbours sit close to *each other*, is not a ball. The
trajectory is perfectly smooth without it and impossible with it. That test is
what makes labels unnecessary — nobody has to say where the ball was, only that
it did not jump ten metres and come back inside a tenth of a second.

**Why all three conditions, and not just the first.** A ball genuinely changes
direction the instant it is kicked, and a detection right after a kick is far
from the frame before it. What separates a kick from a phantom is the third
condition: across a kick the frame before and the frame after are also far
apart, because the ball really did move. Across a phantom they are neighbours.
Dropping that condition would mine every kick in the match and teach the model
to go blind exactly when the ball is struck.

**The threshold sets itself.** How far a ball moves between two frames depends
on the camera, the zoom and the sampling rate, so a fixed pixel distance would
have to be retuned for every video. The scale is taken from the footage itself
— a multiple of the median frame-to-frame displacement — so the same code works
on a tactical wide shot and a tight broadcast angle.

**The trap that ruined the first attempt at this.** A window cropped around a
false positive very often still contains the real ball. Writing it out as
"empty" teaches the model to suppress the thing it exists to find. Here the
real ball's position is not known — that is the whole point — but it can be
*estimated*, by interpolating between the two good neighbours the test already
required. When that estimate falls inside the crop, the crop is written with a
label rather than empty.

    python training/scripts/mine_from_match.py input_videos/psg_inter_10min.mp4
    python training/scripts/mine_from_match.py <video> --write
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CROPS = ROOT / "datasets" / "ball_crops"
WINDOW = 640


@dataclass
class Sighting:
    """One frame's best ball box, or the absence of one."""

    frame: int
    xy: np.ndarray | None
    conf: float = 0.0
    box: np.ndarray | None = None


@dataclass
class Teleport:
    """A sighting the trajectory is better off without."""

    index: int          # position in the sightings list
    frame: int
    xy: np.ndarray
    conf: float
    box: np.ndarray
    expected: np.ndarray   # where the ball actually was, by interpolation
    jump: float            # how far the detection sat from that


def median_step(sightings: Sequence[Sighting]) -> float:
    """Typical frame-to-frame movement, which sets the scale for everything."""
    steps = [
        float(np.linalg.norm(b.xy - a.xy))
        for a, b in zip(sightings, sightings[1:])
        if a.xy is not None and b.xy is not None and b.frame - a.frame <= 2
    ]
    return float(np.median(steps)) if steps else 0.0


def find_teleports(
    sightings: Sequence[Sighting],
    factor: float = 6.0,
    max_gap: int = 3,
    scale: float | None = None,
) -> List[Teleport]:
    """Detections that are far from both neighbours while those two are close.

    `factor` multiplies the footage's own median step to get the distance that
    counts as impossible. Six is deliberately generous: the aim is to collect
    detections that are certainly wrong, not every detection that might be.
    """
    if scale is None:
        scale = median_step(sightings)
    if not scale:
        return []
    limit = factor * scale

    found: List[Teleport] = []
    for i in range(1, len(sightings) - 1):
        current, before, after = sightings[i], sightings[i - 1], sightings[i + 1]
        if current.xy is None or before.xy is None or after.xy is None:
            continue
        # Neighbours must be close in *time* as well, or "the frame before"
        # is half a second ago and the ball is allowed to have moved.
        if current.frame - before.frame > max_gap or after.frame - current.frame > max_gap:
            continue

        in_jump = float(np.linalg.norm(current.xy - before.xy))
        out_jump = float(np.linalg.norm(after.xy - current.xy))
        bridge = float(np.linalg.norm(after.xy - before.xy))

        # Far from both, while the two of them are neighbours: the path is
        # smooth without this detection and impossible with it.
        if in_jump > limit and out_jump > limit and bridge <= limit:
            span = after.frame - before.frame
            t = (current.frame - before.frame) / span if span else 0.5
            expected = before.xy + (after.xy - before.xy) * t
            found.append(Teleport(
                index=i, frame=current.frame, xy=current.xy, conf=current.conf,
                box=current.box, expected=expected,
                jump=float(np.linalg.norm(current.xy - expected)),
            ))
    return found


def watch(video: Path, stride: int, limit: int | None) -> List[Sighting]:
    """Run the ball detector over the video the way the pipeline runs it."""
    import football_ai.video as vid
    from football_ai.detect import Detector, DetectorConfig

    detector = Detector(DetectorConfig())
    sightings: List[Sighting] = []
    seen = 0
    for index, frame in vid.frames(video):
        if index % stride:
            continue
        det = detector.detect_batch([index], [frame])[0]
        centre = det.ball_centre
        sightings.append(Sighting(
            frame=index,
            xy=None if centre is None else np.asarray(centre, dtype=float),
            conf=det.ball_conf,
            box=None if det.ball is None else np.asarray(det.ball, dtype=float),
        ))
        seen += 1
        if seen % 250 == 0:
            found = sum(1 for s in sightings if s.xy is not None)
            print(f"    {seen} frames, ball in {found / len(sightings):.0%}")
        if limit and seen >= limit:
            break
    return sightings


def inside(point: np.ndarray, x0: int, y0: int, margin: int = 8) -> bool:
    return (
        x0 + margin <= point[0] <= x0 + WINDOW - margin
        and y0 + margin <= point[1] <= y0 + WINDOW - margin
    )


def write(video: Path, teleports: Sequence[Teleport], split: str) -> tuple[int, int]:
    """Crop each mistake out of the video. Returns (written, with_ball)."""
    import cv2
    import football_ai.video as vid

    images, labels = CROPS / split / "images", CROPS / split / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    wanted = {t.frame: t for t in teleports}
    written = kept = 0
    for index, frame in vid.frames(video):
        t = wanted.get(index)
        if t is None:
            continue
        height, width = frame.shape[:2]
        x0 = int(np.clip(t.xy[0] - WINDOW / 2, 0, max(width - WINDOW, 0)))
        y0 = int(np.clip(t.xy[1] - WINDOW / 2, 0, max(height - WINDOW, 0)))
        crop = frame[y0:y0 + WINDOW, x0:x0 + WINDOW]
        if crop.shape[0] != WINDOW or crop.shape[1] != WINDOW:
            continue

        # The trap: the real ball is often still in this window. Its position
        # is not labelled anywhere — it is the interpolation the test already
        # produced. Writing the crop empty would train the misses back in.
        label = ""
        if inside(t.expected, x0, y0):
            size = 22.0  # a ball at this scale; the box only has to contain it
            cx = (t.expected[0] - x0) / WINDOW
            cy = (t.expected[1] - y0) / WINDOW
            label = f"0 {cx:.6f} {cy:.6f} {size / WINDOW:.6f} {size / WINDOW:.6f}\n"
            kept += 1

        name = f"match_{video.stem[:20]}_{index:07d}"
        cv2.imwrite(str(images / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (labels / f"{name}.txt").write_text(label)
        written += 1
    return written, kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--stride", type=int, default=2, help="analyse every Nth frame")
    ap.add_argument("--factor", type=float, default=6.0,
                    help="multiples of the footage's median step that count as impossible")
    ap.add_argument("--limit", type=int, help="stop after N analysed frames")
    ap.add_argument("--split", default="train")
    ap.add_argument("--write", action="store_true", help="add the crops to the dataset")
    args = ap.parse_args()

    if not args.video.exists():
        print(f"no such file: {args.video}", file=sys.stderr)
        return 1

    print(f"\n  watching {args.video.name} for impossible balls")
    sightings = watch(args.video, args.stride, args.limit)
    seen = sum(1 for s in sightings if s.xy is not None)
    scale = median_step(sightings)
    print(f"\n  {len(sightings)} frames analysed, ball found in {seen} ({seen / max(len(sightings),1):.0%})")
    print(f"  median step {scale:.1f} px — impossible is beyond {args.factor * scale:.0f} px")

    teleports = find_teleports(sightings, factor=args.factor)
    print(f"\n  {len(teleports)} impossible detections "
          f"({len(teleports) / max(seen, 1):.2%} of sightings)")
    if teleports:
        confs = np.array([t.conf for t in teleports])
        jumps = np.array([t.jump for t in teleports])
        print(f"    confidence  median {np.median(confs):.2f}  max {confs.max():.2f}")
        print(f"    distance from the real path  median {np.median(jumps):.0f} px  "
              f"max {jumps.max():.0f} px")
        print(f"\n    these are detections the model made confidently on something")
        print(f"    that is not a ball, in footage it has never been trained on.")

    if args.write and teleports:
        written, kept = write(args.video, teleports, args.split)
        print(f"\n  wrote {written} crops to {CROPS / args.split}")
        print(f"  {kept} of them still contain the real ball and keep its label —")
        print(f"  labelling those empty would train the misses back in.")
        print(f"\n  retrain with:\n"
              f"    .venv/bin/python training/scripts/train.py ball "
              f"--base models/ball.pt --epochs 12 --batch 8\n")
    elif not args.write:
        print(f"\n  nothing written. Pass --write to add these crops.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
