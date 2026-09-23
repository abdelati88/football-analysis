#!/usr/bin/env bash
# Wait for the players run to finish, train the ball model, keep the better one.
#
# The machine has 16 GB of unified memory and cannot hold two trainings at once
# — attempting it drove swap to 13 of 14 GB and the ball run to 28 seconds per
# step, about fifty hours. So this waits rather than competing.
#
# The ball model is deployed only if it is measurably better than the one it
# would replace, on the regime that actually matters: a native-resolution
# window around the ball's last known position, which is how nearly every frame
# is handled. Improving whole-frame mAP while getting worse there would be a
# regression dressed as progress, so the check is explicit and the previous
# weights are kept either way.
#
#   ./training/scripts/overnight.sh            wait for players, then train ball
#   ./training/scripts/overnight.sh --now      skip the wait
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY=./.venv/bin/python

LOG="$ROOT/training/runs/overnight.log"
BALL_LOG="$ROOT/training/runs/ball_train.log"
BACKUP="$ROOT/training/runs/ball_previous.pt"
BEST="$ROOT/training/runs/ball/weights/best.pt"
EPOCHS="${EPOCHS:-45}"

mkdir -p "$ROOT/training/runs"
say() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

say "=== overnight run starting ==="

# ---- 1. wait for any training already in flight -------------------------
if [ "${1:-}" != "--now" ]; then
  waited=0
  while pgrep -f "train.py players" > /dev/null 2>&1; do
    if [ $((waited % 600)) -eq 0 ]; then
      epoch=$(tail -1 "$ROOT/training/runs/players/results.csv" 2>/dev/null | cut -d, -f1)
      say "players still training (epoch ${epoch:-?}/100) — waiting"
    fi
    sleep 60
    waited=$((waited + 60))
  done
  say "players training has finished after ${waited}s of waiting"
fi

# Give the memory back before asking for it again.
sleep 20

# ---- 2. keep the model we might be replacing ---------------------------
if [ -f "$ROOT/models/ball.pt" ]; then
  cp "$ROOT/models/ball.pt" "$BACKUP"
  say "kept the current ball model at $BACKUP"
fi

# ---- 3. train ----------------------------------------------------------
say "training the ball model on native-resolution windows ($EPOCHS epochs)"
$PY training/scripts/train.py ball --epochs "$EPOCHS" > "$BALL_LOG" 2>&1
status=$?
if [ $status -ne 0 ]; then
  say "TRAINING FAILED (exit $status) — see $BALL_LOG"
  say "the previous ball model is untouched"
  exit $status
fi
say "training finished"

# ---- 4. measure both, on the regimes the detector actually uses --------
say "measuring the new model against the old one"
$PY training/scripts/evaluate_ball.py \
    --compare "$BACKUP" "$ROOT/models/ball.pt" \
    --json "$ROOT/training/runs/ball_comparison.json" \
    >> "$LOG" 2>&1

# ---- 5. deploy only on the number that matters -------------------------
# train.py has already copied best.pt over models/ball.pt, so a regression has
# to be actively undone rather than merely not applied.
$PY - <<'PYEOF' >> "$LOG" 2>&1
import json, shutil, sys
from pathlib import Path

ROOT = Path.cwd()   # the shell script cd'd to the repo root
report = json.loads((ROOT / "training/runs/ball_comparison.json").read_text())
backup = ROOT / "training/runs/ball_previous.pt"
key = "native window (hot path)"

rows = list(report.items())
if len(rows) < 2:
    print("only one model measured — leaving the deployed weights alone")
    sys.exit(0)

(old_path, old), (new_path, new) = rows[0], rows[1]
o, n = old[key], new[key]
print(f"\n  hot path — old: found {o['found']:.0%}, wrong {o['wrong']:.0%}")
print(f"  hot path — new: found {n['found']:.0%}, wrong {n['wrong']:.0%}")

# A wrong box costs more than a miss. When the ball is missing, the
# interpolation bridges the gap and possession survives; when the ball is
# confidently somewhere else, possession is handed to whoever is standing
# there and every event follows it. So the score charges more for a wrong
# box than it credits for a find, and a model that merely finds more while
# being wrong just as often does not automatically win.
#
# Comparing the two rates separately looked reasonable and was not: it would
# have rejected a model that found the same number of balls with a fraction
# of the wrong ones, which is the largest improvement available here.
def score(r):
    return r["found"] - 1.5 * r["wrong"]

better = score(n) > score(o) + 0.02
print(f"  score — old {score(o):+.2f}, new {score(n):+.2f}")
if better:
    print("  the new model is better on both counts — keeping it deployed")
else:
    shutil.copy(backup, ROOT / "models/ball.pt")
    print("  the new model is NOT clearly better — restored the previous weights")
    print("  (the new ones are still at training/runs/ball/weights/best.pt)")
    print("  ACTION NEEDED: the tiled sweep in detect.py suits a crop-trained")
    print("  model and hurts this one. Set DetectorConfig.ball_tiled_sweep")
    print("  to False to go back to the single downscaled pass.")
PYEOF

say "=== overnight run complete — see $LOG and training/runs/ball_comparison.json ==="
