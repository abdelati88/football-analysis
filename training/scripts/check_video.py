"""Is this video worth tagging? Answer in half a minute, not after an hour.

Tagging ten minutes of football by hand costs a person an hour. Discovering
afterwards that the footage was a highlight reel — all close-ups and replays,
with the pitch lines never visible — wastes that hour completely, and the only
way to find out has been to spend it.

So this samples frames across a video and asks the three questions that decide
whether the analysis can work on it at all:

  * **Can the pitch be located?** Every event has a position in metres, and
    metres come from matching the pitch's own lines against a known model. A
    close-up of a player's face contains no lines, so it yields no position. If
    most frames cannot be solved, no event can be placed.
  * **Are the players small?** A tactical wide angle puts twenty-two players in
    frame at a hundred pixels tall. A broadcast close-up puts three in frame at
    six hundred. The second looks better and is worth nothing: there is no team
    shape to measure.
  * **How often does the camera cut?** A montage cutting every two seconds
    never holds a passage of play long enough for possession to mean anything.

None of these need the whole video, so it reads a few dozen frames spread
across it and reports what it found, with a verdict in plain words.

    python training/scripts/check_video.py match.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from football_ai import video as vid  # noqa: E402
from football_ai.calibrate import CalibrationConfig, PitchCalibrator  # noqa: E402
from football_ai.detect import Detector, DetectorConfig, PLAYER  # noqa: E402
from football_ai.pitch import PITCH  # noqa: E402
from football_ai.segments import SegmentConfig, distance, signature  # noqa: E402


def check(path: Path, samples: int) -> dict:
    info = vid.probe(path)
    frames = vid.sample_frames(path, count=samples)
    if not frames:
        raise SystemExit(f"could not read any frames from {path}")

    detector = Detector(DetectorConfig())
    calibrator = PitchCalibrator(CalibrationConfig(), PITCH)

    solved = 0
    heights: list[float] = []
    counts: list[int] = []
    for index, frame in frames:
        if calibrator.solve_frame(index, frame) is not None:
            solved += 1
        det = detector.detect_batch([index], [frame])[0]
        boxes = [det.people.xyxy[j] for j in range(len(det.people))
                 if det.people.class_id[j] == PLAYER]
        counts.append(len(boxes))
        heights.extend(float(b[3] - b[1]) for b in boxes)

    # Cuts: walk consecutive frames in a few short bursts rather than the whole
    # video, which would mean decoding all of it.
    seg = SegmentConfig()
    cuts = 0
    watched = 0.0
    step = max(int(info.frame_count / 6), 1) if info.frame_count else 0
    for start in range(0, max(info.frame_count - 150, 1), max(step, 1)):
        previous = None
        for _, frame in vid.frames(path, start=start, stop=start + 125):
            sig = signature(frame, seg)
            if previous is not None and distance(sig, previous) > seg.min_cut_distance * seg.cut_ratio:
                cuts += 1
            previous = sig
            watched += 1
        if watched > 700:
            break

    return {
        "info": info,
        "solved": solved / len(frames),
        "median_height": float(np.median(heights)) if heights else 0.0,
        "median_players": float(np.median(counts)) if counts else 0.0,
        "cuts_per_minute": cuts / (watched / info.fps / 60) if watched and info.fps else 0.0,
        "sampled": len(frames),
    }


def verdict(r: dict) -> tuple[str, list[str]]:
    """Plain words, and the reasons behind them."""
    reasons = []
    fatal = False

    if r["solved"] < 0.5:
        reasons.append(f"the pitch lines are readable in only {r['solved']:.0%} of frames — "
                       "events cannot be placed on a pitch")
        fatal = True
    elif r["solved"] < 0.8:
        reasons.append(f"the pitch solves in {r['solved']:.0%} of frames; usable but not ideal")
    else:
        reasons.append(f"the pitch solves in {r['solved']:.0%} of frames")

    frac = r["median_height"] / r["info"].height if r["info"].height else 0
    if frac > 0.35:
        reasons.append(f"players fill {frac:.0%} of the frame height — this is a close-up, "
                       "not a tactical angle")
        fatal = True
    else:
        reasons.append(f"players are {frac:.0%} of frame height, {r['median_players']:.0f} "
                       "visible at a time")

    if r["cuts_per_minute"] > 20:
        reasons.append(f"the camera cuts {r['cuts_per_minute']:.0f} times a minute — "
                       "this is a montage, not continuous play")
        fatal = True
    else:
        reasons.append(f"{r['cuts_per_minute']:.0f} camera cuts a minute")

    if fatal:
        return "NOT USABLE", reasons
    if r["solved"] < 0.8 or r["cuts_per_minute"] > 8:
        return "USABLE, but not the best footage to judge the system on", reasons
    return "GOOD — tag this one", reasons


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--samples", type=int, default=24)
    args = ap.parse_args()

    if not args.video.exists():
        print(f"no such file: {args.video}", file=sys.stderr)
        return 1

    print(f"\n  reading {args.video.name} …")
    r = check(args.video, args.samples)
    i = r["info"]
    print(f"\n  {i.width}x{i.height} at {i.fps:.0f} fps, {i.duration_s / 60:.1f} minutes")

    call, reasons = verdict(r)
    print(f"\n  {call}\n")
    for line in reasons:
        print(f"    · {line}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
