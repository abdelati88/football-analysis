"""How stable is the tracking on a given video?

Detection accuracy is measured against a labelled test set; tracking stability
cannot be, because the datasets are single frames. This is the next best thing:
run the real detector and tracker over real footage and count how many
identities they create.

What to look for. `identities` should be close to the number of people who
appear — around 25 for a full pitch, more if the camera pans enough for players
to leave and re-enter. A number several times that means the tracker is losing
players and re-labelling them, which fragments every per-player statistic and
manufactures phantom turnovers where one identity ends and another begins.

By default the pitch model is loaded too, so that the offline re-linking in
`football_ai.stitch` can be measured against the raw tracker — the whole point
of that stage is to close these gaps, and the only honest way to report it is
the same clip counted both ways. Pass `--no-stitch` for the raw figure alone.

    python training/scripts/diagnose_tracking.py input_videos/match.mp4
    python training/scripts/diagnose_tracking.py match.mp4 --seconds 60
    python training/scripts/diagnose_tracking.py match.mp4 --no-stitch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from football_ai import video as vid                                  # noqa: E402
from football_ai.calibrate import (                                   # noqa: E402
    CalibrationConfig, CalibrationTrack, PitchCalibrator,
)
from football_ai.detect import PLAYER, Detector, DetectorConfig       # noqa: E402
from football_ai.stitch import StitchConfig, stitch                   # noqa: E402
from football_ai.teams import TeamClassifier, TeamConfig              # noqa: E402
from football_ai.track import (                                       # noqa: E402
    TrackConfig, TrackStore, foot_position, make_tracker,
    smooth_positions, track_classes,
)


def churn_of(people, verdict: bool = True) -> tuple[float, str]:
    per_frame = people.groupby("frame").size()
    identities = people["track_id"].nunique()
    value = identities / max(per_frame.mean(), 1)
    label = (
        "stable" if value < 1.8 else
        "some fragmentation" if value < 3.0 else
        "heavy fragmentation — per-player statistics will be unreliable"
    )
    return value, label


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--fps", type=float, default=12.5, help="rate to analyse at")
    ap.add_argument("--device")
    ap.add_argument("--no-stitch", action="store_true", help="skip the re-linking comparison")
    args = ap.parse_args()

    if not args.video.exists():
        print(f"No such video: {args.video}", file=sys.stderr)
        return 1

    info = vid.probe(args.video)
    stride = max(int(round(info.fps / args.fps)), 1)
    effective_fps = info.fps / stride

    detector = Detector(DetectorConfig(batch=8, device=args.device))
    tracker = make_tracker(TrackConfig(), effective_fps)
    store = TrackStore()

    # The re-linking reasons in metres, so it needs the homography; and it uses
    # kit colour, so it needs the team classifier. Both are skipped when only
    # the raw tracker figure is wanted, which makes that run much faster.
    calibrator = None
    classifier = None
    if not args.no_stitch:
        try:
            calibrator = PitchCalibrator(CalibrationConfig())
        except FileNotFoundError as exc:
            print(f"  (no pitch model: {exc}) — falling back to image space\n")
        samples = vid.sample_frames(args.video, count=16)
        pairs = []
        for index, frame in samples:
            det = detector.detect_batch([index], [frame])[0]
            boxes = [
                det.people.xyxy[j] for j in range(len(det.people))
                if det.people.class_id[j] == PLAYER
            ]
            if boxes:
                pairs.append((frame, boxes))
        try:
            classifier = TeamClassifier(TeamConfig()).fit(pairs)
        except ValueError as exc:
            print(f"  (kit clustering unavailable: {exc})\n")
            classifier = None
        detector.reset()

    calibration = CalibrationTrack(CalibrationConfig())
    stop = int(args.seconds * info.fps)
    sampled = 0
    for indices, images in vid.batches(args.video, size=8, stop=stop, stride=stride):
        for detection, frame in zip(detector.detect_batch(indices, images), images):
            if calibrator is not None and sampled % CalibrationConfig().every_n_frames == 0:
                calibration.add(calibrator.solve_frame(detection.index, frame))
            homography = calibration.at(detection.index)

            tracked = tracker.update_with_detections(detection.people)
            pitch_xy = None
            if homography is not None and len(tracked):
                feet = np.stack([foot_position(b) for b in tracked.xyxy])
                pitch_xy = homography.to_pitch(feet)
            store.add_people(detection.index, detection.index / info.fps, tracked, pitch_xy)

            if classifier is not None and len(tracked) and tracked.tracker_id is not None:
                classifier.observe(
                    frame,
                    [
                        (int(tid), tracked.xyxy[j])
                        for j, tid in enumerate(tracked.tracker_id)
                        if tid is not None and tracked.class_id[j] == PLAYER
                    ],
                )
            sampled += 1

    people = store.people_frame()
    if people.empty:
        print("Nothing was detected.")
        return 2

    lives = people.groupby("track_id")["frame"].agg(["min", "max", "size"])
    lives["span_s"] = (lives["max"] - lives["min"]) / info.fps
    per_frame = people.groupby("frame").size()

    print(f"\n  {args.video.name} — first {args.seconds:.0f} s at {effective_fps:.1f} fps\n")
    print(f"  frames analysed        {people['frame'].nunique()}")
    print(f"  people per frame       {per_frame.mean():.1f}  (min {per_frame.min()}, max {per_frame.max()})")
    print(f"  identities created     {len(lives)}")
    print(f"  lasting 5 s or more    {(lives.span_s >= 5).sum()}")
    print(f"  median lifetime        {lives.span_s.median():.1f} s")
    print(f"  mean lifetime          {lives.span_s.mean():.1f} s")

    raw_churn, raw_verdict = churn_of(people)
    print(f"\n  identities per person  {raw_churn:.1f}  ->  {raw_verdict}")

    if args.no_stitch:
        print()
        return 0

    smoothed = smooth_positions(people)
    teams = classifier.finalise() if classifier is not None else {}
    kits = classifier.kit_features() if classifier is not None else None
    # info.fps, not effective_fps: `frame` counts original video frames.
    linked, _, report = stitch(
        smoothed, teams, track_classes(smoothed), kits,
        fps=info.fps, frame_width=float(info.width), config=StitchConfig(),
    )

    joined = linked.groupby("track_id")["frame"].agg(["min", "max"])
    joined["span_s"] = (joined["max"] - joined["min"]) / info.fps
    churn, verdict = churn_of(linked)

    print(f"\n  ── after offline re-linking ──────────────────────────────\n")
    print(f"  pitch coverage         {calibration.coverage:.0%}")
    for entry in report.per_round:
        print(f"  merges below {entry['max_gap_s']:>4.1f} s     {entry['merges']}")
    print(f"  identities remaining   {report.tracks_after}   ({report.merges} merged away)")
    print(f"  lasting 5 s or more    {(joined.span_s >= 5).sum()}")
    print(f"  median lifetime        {joined.span_s.median():.1f} s")
    print(f"  mean lifetime          {joined.span_s.mean():.1f} s")
    print(f"\n  identities per person  {churn:.1f}  ->  {verdict}")
    print(f"  improvement            {raw_churn:.1f} -> {churn:.1f}  ({report.reduction:.0%} fewer identities)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
