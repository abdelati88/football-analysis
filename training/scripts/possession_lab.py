"""Re-derive possession and events from cached tracks, and score them.

`cache_tracks.py` froze the detection stage. This reads it back, runs the
possession and event layers over it, and scores the result against hand-tagged
play — the whole loop in about a second, so a change to a threshold can be
measured instead of argued about.

    python training/scripts/cache_tracks.py input_videos/08fd33_4.mp4
    python training/scripts/possession_lab.py --truth evaluation/truth_08fd33_4.json

Any field of PossessionConfig can be overridden on the command line, which is
how a candidate change is tried before it is written into the source:

    python training/scripts/possession_lab.py --truth ... --set confirm_frames=4
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import fields, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from football_ai.events import EventBuilder, EventConfig, infer_attacking_directions  # noqa: E402
from football_ai.pitch import PITCH  # noqa: E402
from football_ai.possession import (  # noqa: E402
    PossessionConfig, assign_ball_holder, build_spells, possession_share,
)

sys.path.insert(0, str(ROOT / "training" / "scripts"))
from evaluate_events import from_file, score  # noqa: E402


def derive(cache: dict, possession: PossessionConfig, events_cfg: EventConfig) -> list[dict]:
    people, ball = cache["people"], cache["ball"]
    teams, classes = cache["teams"], cache["classes"]
    holder = assign_ball_holder(people, ball, teams, classes, possession, cache["width"])
    spells = build_spells(holder, people, ball, teams, classes, possession)
    directions = infer_attacking_directions(people, teams, classes, PITCH)
    builder = EventBuilder(
        team_names={1: "Team 1", 2: "Team 2"}, directions=directions,
        player_names={}, pitch=PITCH, config=events_cfg, half=1, fps=cache["fps"],
    )
    built = builder.build(spells, ball)
    return [e.as_tagger_row() for e in built], holder, spells


def apply_overrides(config, pairs):
    known = {f.name: f.type for f in fields(config)}
    patch = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        if key not in known:
            raise SystemExit(f"unknown setting {key!r}; known: {', '.join(sorted(known))}")
        current = getattr(config, key)
        patch[key] = type(current)(raw) if not isinstance(current, bool) else raw.lower() in ("1", "true", "yes")
    return replace(config, **patch) if patch else config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=ROOT / "evaluation" / "tracks.pkl")
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--tolerance", type=float, default=4.0)
    ap.add_argument("--set", nargs="*", dest="overrides", help="field=value on PossessionConfig")
    ap.add_argument("--spells", action="store_true", help="list the possession spells")
    ap.add_argument("--save", type=Path, help="write the predicted events out")
    args = ap.parse_args()

    cache = pickle.load(args.cache.open("rb"))
    config = apply_overrides(PossessionConfig(), args.overrides)
    rows, holder, spells = derive(cache, config, EventConfig())

    if args.spells:
        print(f"\n  {len(spells)} spells")
        for s in spells:
            print(f"    {s.start_time:5.1f}-{s.end_time:5.1f}  #{s.track_id:3d} team{s.team}  "
                  f"{s.n_frames:3d}f  carry {s.carry_distance:4.1f}m")

    truth = from_file(args.truth)
    kinds = {e["Event"] for e in truth}
    same = [r for r in rows if r["Event"] in kinds]

    for label, predicted in (("all kinds", rows), ("tagged kinds only", same)):
        s = score(truth, predicted, args.tolerance)
        print(f"\n  {label}: {len(predicted)} events")
        print(f"    recall {s['recall']:5.0%}   precision {s['precision']:5.0%}   "
              f"outcome {s['outcome_accuracy']:5.0%}   timing {s['median_time_error_s']:.1f}s")
        if label == "tagged kinds only":
            for e in s["missed_events"]:
                print(f"      missed   {e['Mins']}:{e['Secs']:02d} {e['Event']}/{e['Team']}")
            for e in s["spurious_events"]:
                print(f"      invented {e['Mins']}:{e['Secs']:02d} {e['Event']}/{e['Team']}")

    if args.save:
        json.dump({"events": rows}, args.save.open("w"), indent=2)
        print(f"\n  wrote {args.save}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
