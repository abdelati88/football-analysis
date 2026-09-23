"""Freeze the expensive half of the pipeline so the cheap half can be iterated on.

Detection, tracking, calibration and team classification take a minute on a
thirty-second clip and several on a match. Possession and events are derived
from their output in milliseconds. Tuning possession by re-running the whole
pipeline means a minute per experiment, which is enough friction to make anyone
stop measuring and start guessing — and guessing at this layer is exactly how a
plausible-looking wrong answer survives.

So this runs the pipeline up to the point where the geometry is settled, and
writes what it found to a single file. `possession_lab.py` reads it back and
re-derives from there in under a second.

    python training/scripts/cache_tracks.py input_videos/08fd33_4.mp4
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from football_ai.calibrate import PitchCalibrator  # noqa: E402
from football_ai.detect import Detector, GOALKEEPER, PLAYER  # noqa: E402
from football_ai.pipeline import Pipeline, PipelineConfig  # noqa: E402
from football_ai.stitch import stitch  # noqa: E402
from football_ai.teams import TeamClassifier  # noqa: E402
from football_ai.track import (  # noqa: E402
    drop_short_tracks, interpolate_ball, smooth_positions, track_classes,
)
from football_ai import video as vid  # noqa: E402


def build(video: Path, config: PipelineConfig) -> dict:
    """Everything `_derive` has in hand once identities are settled."""
    pipeline = Pipeline(config, on_progress=lambda s, f, m: print(f"  {s:9s} {f:5.0%}  {m}"))
    info = vid.probe(video)
    target = config.sample_fps or info.fps
    stride = max(int(round(info.fps / target)), 1)
    stop = int(config.max_seconds * info.fps) if config.max_seconds else None

    detector = Detector(config.detector)
    calibrator = PitchCalibrator(config.calibration, config.pitch)
    classifier = pipeline._survey(video, detector)
    store, calibration, _ = pipeline._analyse(
        video, info, detector, calibrator, classifier, stride, stop
    )

    people = smooth_positions(store.people_frame(min_track_length=0))
    ball = interpolate_ball(store.ball_frame(), max_gap=int(info.fps / stride))
    classes = track_classes(people)
    teams = classifier.finalise()

    people, mapping, report = stitch(
        people, teams, classes, classifier.kit_features(),
        fps=info.fps, frame_width=float(info.width), config=config.stitch,
    )
    if report.merges:
        classifier.apply_aliases(mapping)
        teams = classifier.finalise()
        classes = track_classes(people)
    people = drop_short_tracks(people, config.tracking.min_track_length)

    placed = people[people["pitch_x"].notna()]
    if not placed.empty:
        centres = placed.groupby("track_id")[["pitch_x", "pitch_y"]].median()
        keepers = [(int(t), centres.loc[t].to_numpy()) for t in centres.index
                   if classes.get(int(t)) == GOALKEEPER]
        outfield = [(int(t), teams[int(t)], centres.loc[t].to_numpy()) for t in centres.index
                    if classes.get(int(t)) == PLAYER and int(t) in teams]
        teams.update(TeamClassifier.assign_goalkeepers(keepers, outfield))

    return {
        "video": str(video), "people": people, "ball": ball,
        "teams": teams, "classes": classes,
        "fps": info.fps, "width": info.width, "height": info.height,
        "stride": stride,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "evaluation" / "tracks.pkl")
    ap.add_argument("--seconds", type=float)
    args = ap.parse_args()

    config = PipelineConfig(render_video=False)
    if args.seconds:
        config.max_seconds = args.seconds

    started = time.time()
    cache = build(args.video, config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as fh:
        pickle.dump(cache, fh)

    print(f"\n  {len(cache['people'])} player rows, {len(cache['ball'])} ball rows")
    print(f"  {cache['people']['track_id'].nunique()} identities, "
          f"{cache['ball']['pitch_x'].notna().mean():.0%} of ball frames placed on the pitch")
    print(f"  wrote {args.out} in {time.time() - started:.0f}s\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
