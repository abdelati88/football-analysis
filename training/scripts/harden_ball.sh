#!/usr/bin/env bash
# Teach the ball model the mistakes it makes on real match footage.
#
# The dataset the model was trained on is spent — mining it for the model's own
# false positives yields 0.02 per frame, because a model does not get things
# wrong on data it has memorised. On match footage it has never seen, the same
# question yields far more, and `mine_from_match.py` can tell a mistake from a
# ball with no labelling at all: a ball cannot teleport.
#
# This runs the whole thing unattended and, crucially, is reversible at both
# ends. The dataset additions are recorded so they can be removed, and the
# deployed weights are kept so a worse model cannot survive the run — the same
# guard finish_training.sh uses, which has already rejected two models that
# looked like improvements and were not.
#
#   caffeinate -i bash training/scripts/harden_ball.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"
LOG="$ROOT/training/runs/harden.log"
KEEP="$ROOT/training/runs/ball_before_hardening.pt"
MANIFEST="$ROOT/training/runs/mined_crops.txt"
cd "$ROOT"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
mkdir -p "$(dirname "$LOG")"
say "=== hardening the ball model against real match footage ==="

# ---- 1. mine ---------------------------------------------------------------
# Every video the model has never trained on. Each pass writes crops straight
# into the crop dataset; the manifest records exactly which files were added so
# the whole thing can be undone without guessing.

: > "$MANIFEST"
BEFORE=$(ls "$ROOT/datasets/ball_crops/train/images" | wc -l | tr -d ' ')

for VIDEO in "${@:-}"; do
  [ -z "$VIDEO" ] && continue
  say "mining $(basename "$VIDEO")"
  $PY training/scripts/mine_from_match.py "$VIDEO" --stride 2 --write 2>&1 \
    | grep -vE "WARNING|h264|^\[" | tee -a "$LOG"
done

AFTER=$(ls "$ROOT/datasets/ball_crops/train/images" | wc -l | tr -d ' ')
ADDED=$((AFTER - BEFORE))
ls "$ROOT/datasets/ball_crops/train/images" | grep '^match_' > "$MANIFEST" || true
say "added $ADDED crops (dataset now $AFTER); manifest at $MANIFEST"

if [ "$#" -gt 0 ] && [ "$ADDED" -lt 20 ]; then
  say "too few crops to be worth retraining — stopping here, nothing changed"
  exit 0
fi

# ---- 2. train --------------------------------------------------------------
# From the deployed weights rather than from scratch: the point is to correct
# specific mistakes, not to relearn what already works.

cp "$ROOT/models/ball.pt" "$KEEP"
say "kept the deployed ball model at $KEEP"

say "training from models/ball.pt"
$PY training/scripts/train.py ball --base models/ball.pt --epochs 12 --batch 4 \
  >> "$LOG" 2>&1
STATUS=$?
say "training exited $STATUS"

# A crash is not the same as nothing produced. The first run of this hit an
# MPS out-of-memory at epoch 7 of 12, and the checkpoint sitting in the run
# directory had *half* the wrong-box rate of the deployed model — thrown away
# because the script only looked at models/ball.pt, which train.py never got
# to write. Measure whatever was actually produced.
CANDIDATE="$ROOT/models/ball.pt"
RUN_BEST="$ROOT/training/runs/ball/weights/best.pt"
if [ "$STATUS" -ne 0 ]; then
  if [ -f "$RUN_BEST" ] && [ "$RUN_BEST" -nt "$KEEP" ]; then
    say "training crashed but left a checkpoint — measuring it anyway"
    CANDIDATE="$RUN_BEST"
  else
    say "training crashed with no usable checkpoint — nothing to measure"
    say "=== done — full log at $LOG ==="
    exit 0
  fi
fi

# ---- 3. measure, and only then deploy --------------------------------------
# train.py copies its best checkpoint over models/ball.pt, so at this point the
# candidate is deployed and the previous one is in $KEEP. Measure both in the
# regime the pipeline actually uses and put the better one back.

say "measuring the candidate against what was deployed"
# Both paths go to --compare: it *replaces* --weights rather than adding to
# it, so passing one of them as --weights measures a single model and the
# guard below reads that as "only one model measured" and changes nothing.
$PY training/scripts/evaluate_ball.py \
  --compare "$CANDIDATE" "$KEEP" \
  --json "$ROOT/training/runs/harden_comparison.json" 2>&1 \
  | grep -vE "WARNING|h264" | tee -a "$LOG"

$PY - "$ROOT" "$CANDIDATE" <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, shutil, sys
from pathlib import Path

root = Path(sys.argv[1])
report = json.loads((root / "training/runs/harden_comparison.json").read_text())
key = "native window (hot path)"

# A wrong box costs more than a miss: a missing ball is bridged by the
# interpolation, a misplaced one hands possession to the wrong player and
# every event built on it is wrong.
def score(r):
    return r[key]["found"] - 1.5 * r[key]["wrong"]

candidate = sys.argv[2]
previous = str(root / "training/runs/ball_before_hardening.pt")
if candidate not in report or previous not in report:
    print("  only one model measured — leaving the deployed weights alone")
    raise SystemExit(0)

new, old = report[candidate], report[previous]
print(f"  hot path — before: found {old[key]['found']:.0%} wrong {old[key]['wrong']:.0%}")
print(f"  hot path — after:  found {new[key]['found']:.0%} wrong {new[key]['wrong']:.0%}")
print(f"  score — before {score(old):+.3f}, after {score(new):+.3f}")

if score(new) > score(old) + 0.02:
    if candidate != str(root / "models/ball.pt"):
        shutil.copy(candidate, root / "models/ball.pt")
    print("  the hardened model is better — deployed it")
else:
    shutil.copy(previous, candidate)
    print("  the hardened model is not better — restored the previous weights")
    print("  the mined crops stay in the dataset; the manifest lists them")
PYEOF

say "=== done — full log at $LOG ==="
