"""How accurate are the *events*, measured against a person's own tagging.

Every other number in this project measures detection: the ball is found in 94%
of frames, players score 0.987 mAP50, the pitch solves in every frame. None of
them answers the only question a customer actually asks, which is whether the
passes are passes.

They cannot answer it, either. A detector can be perfect while the football is
wrong — the ball is in the right place and possession is still handed to the
nearest player rather than the one who touched it, or a turnover is called an
interception when it was a tackle. Detection accuracy is a precondition for
event accuracy and no kind of evidence for it.

There is no labelled dataset for this and there does not need to be, because
the tagger in this repository is exactly the instrument for making one. Tag a
passage of play by hand, run the analysis over the same passage, and compare.

    # 1. tag ten minutes in the app, save the match, note its id
    # 2. export what the analysis produced for the same clip
    python training/scripts/evaluate_events.py --truth 3 --predicted run.json

    # or compare two exported files directly
    python training/scripts/evaluate_events.py --truth hand.json --predicted ai.json

**How events are matched.** An event is a kind, a team, a moment and a place,
and a person tagging live is a second or two behind what they saw. So a
predicted event matches a hand-tagged one when they agree on kind and team and
fall within a tolerance in time — the default is generous for that reason.
Matching is done as a global assignment rather than greedily, so a burst of
passes a few seconds apart cannot have its first prediction absorb the wrong
one and cascade.

**What is reported.**

    recall      of the events that happened, how many were found
    precision   of the events reported, how many were real
    timing      how far off the matched ones were, which is mostly the
                tagger's own reaction time and is worth seeing separately
    outcome     of the matched events, how many agreed on successful/failed

Recall and precision are reported per event kind as well as overall, because
they are not uniform and the difference matters commercially: passes are the
volume, shots are what gets watched, and a system that is strong on one and
weak on the other should say so rather than hide behind an average.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "match_tag" / "instance" / "football1.db"


# ------------------------------------------------------------------ loading


def _seconds(row: dict) -> float:
    return float(row.get("Mins", 0) or 0) * 60 + float(row.get("Secs", 0) or 0)


def from_file(path: Path) -> List[dict]:
    """Events from an export — either a bare list or {"events": [...]}."""
    blob = json.loads(Path(path).read_text())
    events = blob.get("events", blob) if isinstance(blob, dict) else blob
    return [dict(e) for e in events]


def from_match(match_id: int, db: Path = DB) -> List[dict]:
    """Events saved against one match in the tagger's database."""
    if not db.exists():
        raise FileNotFoundError(f"no database at {db}")
    connection = sqlite3.connect(db)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        "SELECT * FROM events WHERE match_id = ? ORDER BY mins, secs", (match_id,)
    ).fetchall()
    connection.close()
    if not rows:
        raise SystemExit(f"match {match_id} has no events saved")
    return [
        {
            "Team": r["team"], "Player": r["player"], "Event": r["event"],
            "Outcome": r["outcome"], "Mins": r["mins"], "Secs": r["secs"],
            "X": r["x"], "Y": r["y"], "source": r["source"],
        }
        for r in rows
    ]


def hand_tagged_only(events: Sequence[dict]) -> List[dict]:
    """Drop anything an analysis put there.

    A match may hold both — the app imports AI events alongside hand-tagged
    ones — and scoring a prediction against a copy of itself would report a
    perfect system.
    """
    return [e for e in events if (e.get("source") or "manual") != "ai"]


# ------------------------------------------------------------------ matching


def match_events(
    truth: Sequence[dict], predicted: Sequence[dict], tolerance: float,
) -> tuple[List[tuple[int, int]], List[int], List[int]]:
    """Pair predictions with hand-tagged events.

    Returns `(pairs, missed, spurious)` as indices. A pair must agree on kind
    and team and lie within `tolerance` seconds; among the pairs that qualify,
    the assignment minimising total time error is chosen, so neighbouring
    events of the same kind cannot be mismatched into a cascade.
    """
    if not truth or not predicted:
        return [], list(range(len(truth))), list(range(len(predicted)))

    LARGE = 1e6
    cost = np.full((len(truth), len(predicted)), LARGE)
    allowed = np.zeros_like(cost, dtype=bool)

    for i, t in enumerate(truth):
        for j, p in enumerate(predicted):
            if str(t.get("Event", "")).lower() != str(p.get("Event", "")).lower():
                continue
            if str(t.get("Team", "")).strip() != str(p.get("Team", "")).strip():
                continue
            gap = abs(_seconds(t) - _seconds(p))
            if gap > tolerance:
                continue
            cost[i, j] = gap
            allowed[i, j] = True

    rows, cols = linear_sum_assignment(cost)
    pairs = [(int(i), int(j)) for i, j in zip(rows, cols) if allowed[i, j]]
    matched_t = {i for i, _ in pairs}
    matched_p = {j for _, j in pairs}
    return (
        pairs,
        [i for i in range(len(truth)) if i not in matched_t],
        [j for j in range(len(predicted)) if j not in matched_p],
    )


def score(
    truth: Sequence[dict], predicted: Sequence[dict], tolerance: float,
) -> dict:
    pairs, missed, spurious = match_events(truth, predicted, tolerance)

    gaps = [abs(_seconds(truth[i]) - _seconds(predicted[j])) for i, j in pairs]
    outcome_agreed = sum(
        1 for i, j in pairs
        if str(truth[i].get("Outcome") or "").lower()
        == str(predicted[j].get("Outcome") or "").lower()
    )

    per_kind: Dict[str, dict] = defaultdict(
        lambda: {"truth": 0, "predicted": 0, "matched": 0}
    )
    for e in truth:
        per_kind[str(e.get("Event", "?"))]["truth"] += 1
    for e in predicted:
        per_kind[str(e.get("Event", "?"))]["predicted"] += 1
    for i, _ in pairs:
        per_kind[str(truth[i].get("Event", "?"))]["matched"] += 1

    return {
        "truth": len(truth),
        "predicted": len(predicted),
        "matched": len(pairs),
        "missed": len(missed),
        "spurious": len(spurious),
        "recall": len(pairs) / len(truth) if truth else 0.0,
        "precision": len(pairs) / len(predicted) if predicted else 0.0,
        "outcome_accuracy": outcome_agreed / len(pairs) if pairs else 0.0,
        "median_time_error_s": float(np.median(gaps)) if gaps else 0.0,
        "per_kind": dict(per_kind),
        "missed_events": [truth[i] for i in missed],
        "spurious_events": [predicted[j] for j in spurious],
    }


# ------------------------------------------------------------------ output


def report(result: dict, tolerance: float) -> None:
    print(f"\n  {result['truth']} hand-tagged events, "
          f"{result['predicted']} from the analysis, "
          f"matched within {tolerance:.0f} s\n")
    print(f"  recall            {result['recall']:.0%}   "
          f"({result['matched']} of {result['truth']} found)")
    print(f"  precision         {result['precision']:.0%}   "
          f"({result['spurious']} reported that were not tagged)")
    print(f"  outcome agreed    {result['outcome_accuracy']:.0%}   "
          f"of the matched events")
    print(f"  median timing     {result['median_time_error_s']:.1f} s off")

    print(f"\n  {'event':<14}{'tagged':>8}{'found':>8}{'reported':>10}"
          f"{'recall':>9}{'precision':>11}")
    for kind, k in sorted(result["per_kind"].items(),
                          key=lambda kv: -kv[1]["truth"]):
        rec = k["matched"] / k["truth"] if k["truth"] else 0.0
        prec = k["matched"] / k["predicted"] if k["predicted"] else 0.0
        print(f"  {kind:<14}{k['truth']:>8}{k['matched']:>8}{k['predicted']:>10}"
              f"{rec:>8.0%}{prec:>10.0%}")

    if result["missed_events"]:
        print(f"\n  missed ({len(result['missed_events'])}) — the analysis said nothing here:")
        for e in result["missed_events"][:8]:
            print(f"    {e.get('Mins',0)}:{int(e.get('Secs',0)):02d}  "
                  f"{e.get('Event')} · {e.get('Outcome') or '—'} · {e.get('Team')}")
        if len(result["missed_events"]) > 8:
            print(f"    … and {len(result['missed_events']) - 8} more")

    if result["spurious_events"]:
        print(f"\n  invented ({len(result['spurious_events'])}) — reported but not tagged:")
        for e in result["spurious_events"][:8]:
            print(f"    {e.get('Mins',0)}:{int(e.get('Secs',0)):02d}  "
                  f"{e.get('Event')} · {e.get('Outcome') or '—'} · {e.get('Team')}")
        if len(result["spurious_events"]) > 8:
            print(f"    … and {len(result['spurious_events']) - 8} more")

    print("\n  Read the two lists before the two numbers. A miss and an invention")
    print("  at the same second are usually one event whose kind was called")
    print("  differently, which is a smaller problem than either figure suggests.\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True,
                    help="a saved match id in the tagger, or a path to an export")
    ap.add_argument("--predicted", required=True, type=Path,
                    help="the analysis output to score (JSON)")
    ap.add_argument("--tolerance", type=float, default=4.0,
                    help="seconds either way a match may be found in. Generous by "
                         "default: a person tagging live is a beat behind the play.")
    ap.add_argument("--json", type=Path, dest="json_out")
    args = ap.parse_args()

    try:
        truth = (
            from_match(int(args.truth)) if str(args.truth).isdigit()
            else from_file(Path(args.truth))
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    truth = hand_tagged_only(truth)
    if not truth:
        print("Nothing hand-tagged to compare against — the events are all from "
              "an analysis, and scoring those against themselves proves nothing.",
              file=sys.stderr)
        return 1

    if not args.predicted.exists():
        print(f"No such file: {args.predicted}", file=sys.stderr)
        return 1
    predicted = from_file(args.predicted)

    result = score(truth, predicted, args.tolerance)
    report(result, args.tolerance)

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {k: v for k, v in result.items()
             if k not in ("missed_events", "spurious_events")}, indent=2))
        print(f"  wrote {args.json_out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
