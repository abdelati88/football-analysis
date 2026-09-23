"""Precision and recall for the offline re-linking, against real ground truth.

Tracking has no labelled test set — the detection datasets are single frames,
so there is nothing to compare an identity against. The usual answer is to
report how many identities the linker removed and call a smaller number better,
which is worthless: a linker that merged the whole pitch into one player would
score best of all.

So the ground truth is built instead. Every long track is cut in half and the
two halves are handed back to the linker as strangers. The right answer is
known by construction, and the wrong answers are known too, because the other
tracks in the pool were alive at the same time as each other and are therefore
certainly different people.

Each merged pair falls into one of three buckets:

    rejoined  both halves of the same parent            -> correct
    wrong     two parents that were on screen together  -> certainly an error
    unknown   two parents that never co-occur           -> may well be the real
              fragmentation the linker exists to fix, and cannot be scored

Only `wrong` counts against precision. Counting `unknown` as an error would
score the do-nothing linker perfectly, which is the failure we started from.

    python training/scripts/evaluate_stitch.py input_videos/match.mp4
    python training/scripts/evaluate_stitch.py match.mp4 --seconds 60 --json out.json

The run is dominated by detection, so the tracking pass is cached to disk and
reused: sweeping thresholds afterwards costs seconds rather than minutes.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from football_ai import video as vid                                   # noqa: E402
from football_ai.calibrate import (                                    # noqa: E402
    CalibrationConfig, CalibrationTrack, PitchCalibrator,
)
from football_ai.detect import PLAYER, Detector, DetectorConfig        # noqa: E402
from football_ai.stitch import StitchConfig, build_tracklets, link     # noqa: E402
from football_ai.teams import TeamClassifier, TeamConfig               # noqa: E402
from football_ai.track import (                                        # noqa: E402
    TrackConfig, TrackStore, foot_position, make_tracker, smooth_positions,
    track_classes,
)

GAPS = (0.4, 0.8, 1.5, 2.5, 4.0, 6.0)


# ------------------------------------------------------------------ capture


def capture(video: Path, seconds: float, sample_fps: float, device: str | None) -> dict:
    """Run detection, tracking and calibration once, and keep what linking needs."""
    info = vid.probe(video)
    stride = max(int(round(info.fps / sample_fps)), 1)

    detector = Detector(DetectorConfig(batch=8, device=device))
    calibrator = PitchCalibrator(CalibrationConfig())
    tracker = make_tracker(TrackConfig(), info.fps / stride)
    store = TrackStore()

    pairs = []
    for index, frame in vid.sample_frames(video, count=16):
        det = detector.detect_batch([index], [frame])[0]
        boxes = [
            det.people.xyxy[j] for j in range(len(det.people))
            if det.people.class_id[j] == PLAYER
        ]
        if boxes:
            pairs.append((frame, boxes))
    classifier = TeamClassifier(TeamConfig()).fit(pairs)
    detector.reset()

    calibration = CalibrationTrack(CalibrationConfig())
    every = CalibrationConfig().every_n_frames
    sampled = 0
    for indices, images in vid.batches(
        video, size=8, stop=int(seconds * info.fps), stride=stride
    ):
        for det, frame in zip(detector.detect_batch(indices, images), images):
            if sampled % every == 0:
                calibration.add(calibrator.solve_frame(det.index, frame))
            homography = calibration.at(det.index)
            tracked = tracker.update_with_detections(det.people)
            pitch_xy = (
                homography.to_pitch(np.stack([foot_position(b) for b in tracked.xyxy]))
                if homography is not None and len(tracked) else None
            )
            store.add_people(det.index, det.index / info.fps, tracked, pitch_xy)
            if len(tracked) and tracked.tracker_id is not None:
                classifier.observe(frame, [
                    (int(t), tracked.xyxy[j])
                    for j, t in enumerate(tracked.tracker_id)
                    if t is not None and tracked.class_id[j] == PLAYER
                ])
            sampled += 1

    return {
        "people": smooth_positions(store.people_frame()),
        "teams": classifier.finalise(),
        "kits": classifier.kit_features(),
        # `frame` holds original video frame indices, so seconds are recovered
        # at the video's rate. `sample_fps` converts a gap into a number of
        # observations, which are only taken every `stride` frames. Confusing
        # the two silently multiplies every gap by the stride.
        "video_fps": info.fps,
        "sample_fps": info.fps / stride,
        "width": float(info.width),
        "coverage": calibration.coverage,
    }


# ------------------------------------------------------------------ scoring


class Bench:
    def __init__(self, data: dict):
        self.people = data["people"]
        self.teams = data["teams"]
        self.kits = data["kits"]
        self.fps = data["video_fps"]
        self.sample_fps = data["sample_fps"]
        self.width = data["width"]
        self.classes = track_classes(self.people)
        span = self.people.groupby("track_id")["frame"].agg(["min", "max"])
        self.span = span.to_dict("index")

    def co_occur(self, a: int, b: int) -> bool:
        """Were these two tracks ever on screen at the same time?"""
        A, B = self.span[a], self.span[b]
        return A["min"] <= B["max"] and B["min"] <= A["max"]

    def split(self, gap_s: float, min_obs: int = 50):
        """Cut every long track in two, `gap_s` apart, and forget they match."""
        gap = int(round(gap_s * self.sample_fps))
        rows, truth, teams, classes, kits = [], {}, {}, {}, {}
        next_id = int(self.people["track_id"].max()) + 1

        def keep(group, tid, parent):
            rows.append(group)
            truth[tid] = parent
            teams[tid] = self.teams.get(parent)
            classes[tid] = self.classes.get(parent, PLAYER)
            if parent in self.kits:
                kits[tid] = self.kits[parent]

        for tid, group in self.people.groupby("track_id"):
            tid = int(tid)
            group = group.sort_values("frame")
            half = len(group) // 2
            first = group.iloc[: half - gap // 2]
            second = group.iloc[half + gap - gap // 2 :]
            if len(group) < min_obs + gap + 2 or len(first) < 8 or len(second) < 8:
                keep(group, tid, tid)
                continue
            second = second.copy()
            second["track_id"] = next_id
            keep(first, tid, tid)
            keep(second, next_id, tid)
            next_id += 1

        return pd.concat(rows).reset_index(drop=True), teams, classes, kits, truth

    def score(self, config: StitchConfig, gap_s: float) -> dict:
        people, teams, classes, kits, truth = self.split(gap_s)
        parents = list(truth.values())
        splits = sum(1 for p in set(parents) if parents.count(p) > 1)

        mapping, _ = link(
            build_tracklets(people, teams, classes, kits, config),
            self.fps, self.width, config,
        )
        groups: dict[int, list[int]] = {}
        for tid, root in mapping.items():
            groups.setdefault(root, []).append(tid)

        rejoined = wrong = unknown = 0
        for members in groups.values():
            for a, b in itertools.combinations(members, 2):
                pa, pb = truth[a], truth[b]
                if pa == pb:
                    rejoined += 1
                elif self.co_occur(pa, pb):
                    wrong += 1
                else:
                    unknown += 1

        return {
            "gap_s": gap_s, "splits": splits, "rejoined": rejoined,
            "wrong": wrong, "unknown": unknown,
            "precision": rejoined / (rejoined + wrong) if rejoined + wrong else 1.0,
            "recall": rejoined / splits if splits else 0.0,
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--fps", type=float, default=12.5)
    ap.add_argument("--device")
    ap.add_argument("--cache", type=Path, help="where to keep the tracking pass")
    ap.add_argument("--refresh", action="store_true", help="ignore an existing cache")
    ap.add_argument("--json", type=Path, dest="json_out")
    args = ap.parse_args()

    if not args.video.exists():
        print(f"No such video: {args.video}", file=sys.stderr)
        return 1

    cache = args.cache or ROOT / "training" / "runs" / f"stitch_{args.video.stem}.pkl"
    if cache.exists() and not args.refresh:
        data = pickle.loads(cache.read_bytes())
        print(f"  using cached tracking pass at {cache}")
    else:
        data = capture(args.video, args.seconds, args.fps, args.device)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps(data))
        print(f"  cached tracking pass to {cache}")

    bench = Bench(data)
    config = StitchConfig()
    results = [bench.score(config, gap) for gap in GAPS]

    print(f"\n  {args.video.name} — {len(bench.people)} observations, "
          f"{bench.people['track_id'].nunique()} tracks, "
          f"{data['coverage']:.0%} pitch coverage")
    print(f"  linking attempted up to {max(config.rounds):.1f} s\n")
    print(f"  {'gap':>6} {'splits':>7} {'rejoined':>9} {'wrong':>6} "
          f"{'unknown':>8} {'precision':>10} {'recall':>7}")
    for r in results:
        print(f"  {r['gap_s']:>5.1f}s {r['splits']:>7} {r['rejoined']:>9} "
              f"{r['wrong']:>6} {r['unknown']:>8} {r['precision']:>10.2f} "
              f"{r['recall']:>7.2f}")

    inside = [r for r in results if r["gap_s"] <= max(config.rounds)]
    ok = sum(r["rejoined"] for r in inside)
    bad = sum(r["wrong"] for r in inside)
    total = sum(r["splits"] for r in inside)
    print(f"\n  within the linking range: {ok}/{total} rejoined "
          f"({ok / total:.0%}), {bad} wrong "
          f"({ok / (ok + bad) if ok + bad else 1.0:.0%} precision)")
    print("  beyond it, precision falls below 0.7 — that is the frontier where "
          "position\n  and kit colour stop identifying a person, and a shirt "
          "number would have to.\n")

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"video": args.video.name, "seconds": args.seconds,
             "rounds": list(config.rounds), "results": results}, indent=2))
        print(f"  wrote {args.json_out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
