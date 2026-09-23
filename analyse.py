#!/usr/bin/env python3
"""Analyse a football video from the command line.

    python analyse.py input_videos/match.mp4
    python analyse.py match.mp4 --team1 Barcelona --team2 "Real Madrid" --limit 120
    python analyse.py match.mp4 --no-video --json out/events.json

Produces the same events and statistics the web app shows — this is the same
pipeline, driven without a browser, which is what makes it usable in a notebook
or a batch job.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from football_ai.pipeline import PipelineConfig, analyse


class ProgressBar:
    """A single line that rewrites itself, rather than a wall of scrollback."""

    STAGES = {
        "survey": "reading kit colours",
        "analyse": "tracking",
        "derive": "finding events",
        "render": "drawing video",
    }

    def __init__(self, width: int = 34):
        self.width = width
        self.started = time.time()
        self.last = 0.0

    def __call__(self, stage: str, fraction: float, message: str) -> None:
        now = time.time()
        if fraction < 0.999 and now - self.last < 0.25:
            return
        self.last = now
        filled = int(self.width * fraction)
        bar = "█" * filled + "·" * (self.width - filled)
        label = self.STAGES.get(stage, stage)
        elapsed = int(now - self.started)
        line = f"\r  {label:<20} [{bar}] {fraction * 100:5.1f}%  {elapsed // 60:02d}:{elapsed % 60:02d}  {message[:44]:<44}"
        sys.stdout.write(line)
        sys.stdout.flush()
        if fraction >= 0.999:
            sys.stdout.write("\n")


def print_report(result) -> None:
    stats = result.stats
    teams = stats.get("teams", {})
    t1 = teams.get(1) or teams.get("1") or {}
    t2 = teams.get(2) or teams.get("2") or {}

    def line(label: str, a, b) -> None:
        print(f"  {str(a):>12}  {label:^26}  {str(b):<12}")

    print("\n" + "=" * 66)
    print(f"  {t1.get('name', 'Team 1'):>12}  {'MATCH REPORT':^26}  {t2.get('name', 'Team 2'):<12}")
    print("=" * 66)
    line("goals", t1.get("goals", 0), t2.get("goals", 0))
    line("possession %", t1.get("possession", 0), t2.get("possession", 0))
    line("passes", t1.get("passes", 0), t2.get("passes", 0))
    line("pass accuracy %", t1.get("pass_accuracy", 0), t2.get("pass_accuracy", 0))
    line("shots", t1.get("shots", 0), t2.get("shots", 0))
    line("shots on target", t1.get("shots_on_target", 0), t2.get("shots_on_target", 0))
    line("crosses", t1.get("crosses", 0), t2.get("crosses", 0))
    line("dribbles", t1.get("dribbles", 0), t2.get("dribbles", 0))
    line("tackles", t1.get("tackles", 0), t2.get("tackles", 0))
    line("interceptions", t1.get("interceptions", 0), t2.get("interceptions", 0))
    line("distance km", t1.get("distance_km", 0), t2.get("distance_km", 0))
    print("=" * 66)

    players = stats.get("players", [])[:12]
    if players:
        print("\n  Most involved players")
        print(f"  {'player':<10}{'team':<6}{'touches':>8}{'passes':>8}{'acc%':>7}{'shots':>7}{'km':>7}{'top km/h':>10}")
        for p in players:
            print(
                f"  {p['name']:<10}{p['team']:<6}{p['touches']:>8}{p['passes']:>8}"
                f"{p['pass_accuracy']:>7}{p['shots']:>7}{p['distance_km']:>7}{p['top_speed_kmh']:>10}"
            )

    diagnostics = result.diagnostics
    calibration = diagnostics.get("calibration", {})
    print(
        f"\n  Pitch located in {calibration.get('solved', 0)} frames "
        f"({calibration.get('coverage', 0) * 100:.0f}% of attempts), "
        f"mean error {calibration.get('mean_error_px', '—')} px"
    )
    print(f"  Ran on {diagnostics.get('device')} in {diagnostics.get('runtime_s')} s")
    if result.video_path:
        print(f"  Annotated video: {result.video_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--team1", default="Team 1")
    ap.add_argument("--team2", default="Team 2")
    ap.add_argument("--half", type=int, default=1, choices=(1, 2))
    ap.add_argument("--limit", type=float, metavar="SECONDS",
                    help="analyse only the first N seconds")
    ap.add_argument("--fps", type=float, default=12.5, metavar="RATE",
                    help="frames per second to analyse (default 12.5)")
    ap.add_argument("--no-video", action="store_true", help="skip the annotated video")
    ap.add_argument("--out", type=Path, default=Path("output_videos"))
    ap.add_argument("--json", type=Path, help="write events and statistics here")
    ap.add_argument("--csv", type=Path, help="write the events as CSV here")
    ap.add_argument("--device", help="cuda, mps or cpu (default: whatever is fastest)")
    args = ap.parse_args()

    if not args.video.exists():
        print(f"No such video: {args.video}", file=sys.stderr)
        return 1

    config = PipelineConfig(
        team_names={1: args.team1, 2: args.team2},
        half=args.half,
        max_seconds=args.limit,
        sample_fps=args.fps,
        render_video=not args.no_video,
    )
    if args.device:
        config.detector.device = args.device
        config.calibration.device = args.device

    print(f"\n  Analysing {args.video.name}\n")
    try:
        result = analyse(args.video, config=config, on_progress=ProgressBar(), output_dir=args.out)
    except FileNotFoundError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2

    print_report(result)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "events": result.tagger_rows(),
            "stats": result.stats,
            "diagnostics": result.diagnostics,
        }, indent=2))
        print(f"  Wrote {args.json}")

    if args.csv:
        import csv as csv_module

        rows = result.tagger_rows()
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            fields = ["Team", "Player", "Event", "Outcome", "Mins", "Secs",
                      "X", "Y", "X2", "Y2", "half", "confidence"]
            writer = csv_module.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Wrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
