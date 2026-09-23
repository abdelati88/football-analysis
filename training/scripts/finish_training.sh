#!/usr/bin/env bash
# Finish both models, one after the other, and never deploy a worse one.
#
# Two phases, sequential because 16 GB of unified memory cannot hold two
# trainings — running them together drove swap to 13 of 14 GB and slowed each
# step from 1.5 seconds to 28.
#
#   1. ball     a longer run than the eight epochs that survived the NaN
#               divergence, this time with AMP off, which is what caused it.
#   2. players  resume from epoch 68 and finish the remaining epochs.
#
# Each phase ends by measuring the new weights against the ones already
# deployed and keeping whichever is better. train.py copies best.pt over
# models/ at the end of a successful run, so a regression has to be actively
# undone rather than merely not applied — that undo is the point of this script.
#
#   screen -dmS finish bash -c 'caffeinate -i ./training/scripts/finish_training.sh'
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY=./.venv/bin/python

LOG="$ROOT/training/runs/finish.log"
# Both models continue from the best weights their interrupted runs reached,
# as fresh runs rather than --resume. Resuming would restore the dead run's own
# hyper-parameters, including the dataloader workers and batch size it was
# holding when the machine killed it — the settings that need changing are
# exactly the ones --resume brings back.
BALL_EPOCHS="${BALL_EPOCHS:-12}"
BALL_BATCH="${BALL_BATCH:-8}"
BALL_FROM="${BALL_FROM:-$ROOT/training/runs/ball/weights/best.pt}"
PLAYERS_EPOCHS="${PLAYERS_EPOCHS:-20}"
PLAYERS_FROM="${PLAYERS_FROM:-$ROOT/training/runs/players/weights/best.pt}"
BALL_BEST="$ROOT/training/runs/ball/weights/best.pt"

mkdir -p "$ROOT/training/runs"
say() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

say "=== finishing both models ==="

# ---------------------------------------------------------------- phase 1
BALL_KEEP="$ROOT/training/runs/ball_deployed_before.pt"
cp "$ROOT/models/ball.pt" "$BALL_KEEP"
say "kept the deployed ball model at $BALL_KEEP"

say "phase 1/2 — ball, $BALL_EPOCHS epochs from $(basename "$BALL_FROM"), batch $BALL_BATCH"
$PY training/scripts/train.py ball --epochs "$BALL_EPOCHS" --batch "$BALL_BATCH" \
    --base "$BALL_FROM" \
    > "$ROOT/training/runs/ball_train.log" 2>&1
status=$?
if [ $status -ne 0 ] && [ -f "$BALL_BEST" ]; then
  # An interrupted run is not an empty one. Ultralytics writes best.pt after
  # every epoch, so a process killed at epoch 21 of 35 — which is what happened
  # here, on memory — still leaves the best weights it reached. Treating a
  # non-zero exit as "nothing to see" threw away a model that measured better
  # than the one deployed, and it took a person noticing to recover it.
  #
  # So: no checkpoint means failure; a checkpoint means measure it and let the
  # numbers decide, exactly as a clean finish would.
  say "training exited badly (status $status) but left a checkpoint — measuring it anyway"
  cp "$BALL_BEST" "$ROOT/models/ball.pt"
  status=0
fi
if [ $status -ne 0 ]; then
  say "ball training failed with no usable checkpoint — restoring the deployed weights"
  cp "$BALL_KEEP" "$ROOT/models/ball.pt"
else
  say "ball training finished — measuring against what was deployed"
  $PY training/scripts/evaluate_ball.py \
      --compare "$BALL_KEEP" "$ROOT/models/ball.pt" \
      --json "$ROOT/training/runs/ball_comparison.json" >> "$LOG" 2>&1

  $PY - <<'PYEOF' >> "$LOG" 2>&1
import json, shutil
from pathlib import Path

ROOT = Path.cwd()
rows = list(json.loads(
    (ROOT / "training/runs/ball_comparison.json").read_text()).items())
key = "native window (hot path)"

# A wrong box costs more than a miss: a missing ball is bridged by the
# interpolation, a misplaced one hands possession to the wrong player and every
# event follows it there.
def score(r):
    return r[key]["found"] - 1.5 * r[key]["wrong"]

if len(rows) < 2:
    print("  only one model measured — leaving the deployed weights alone")
else:
    (_, old), (_, new) = rows
    print(f"  hot path — old: found {old[key]['found']:.0%} wrong {old[key]['wrong']:.0%}")
    print(f"  hot path — new: found {new[key]['found']:.0%} wrong {new[key]['wrong']:.0%}")
    print(f"  score — old {score(old):+.2f}, new {score(new):+.2f}")
    if score(new) > score(old) + 0.02:
        print("  new ball model is better — keeping it")
    else:
        shutil.copy(ROOT / "training/runs/ball_deployed_before.pt",
                    ROOT / "models/ball.pt")
        print("  new ball model is not better — restored the previous weights")
PYEOF
fi
say "phase 1 complete"

sleep 20

# ---------------------------------------------------------------- phase 2
PLAYERS_KEEP="$ROOT/training/runs/players_deployed_before.pt"
cp "$ROOT/models/players.pt" "$PLAYERS_KEEP"
say "kept the deployed players model at $PLAYERS_KEEP"

say "phase 2/2 — players, $PLAYERS_EPOCHS epochs from $(basename "$PLAYERS_FROM")"
$PY training/scripts/train.py players --epochs "$PLAYERS_EPOCHS" \
    --base "$PLAYERS_FROM" \
    > "$ROOT/training/runs/players_train.log" 2>&1
status=$?
PLAYERS_BEST="$ROOT/training/runs/players/weights/best.pt"
if [ $status -ne 0 ] && [ -f "$PLAYERS_BEST" ]; then
  say "training exited badly (status $status) but left a checkpoint — measuring it anyway"
  cp "$PLAYERS_BEST" "$ROOT/models/players.pt"
  status=0
fi
if [ $status -ne 0 ]; then
  say "players training failed with no usable checkpoint — restoring the deployed weights"
  cp "$PLAYERS_KEEP" "$ROOT/models/players.pt"
else
  say "players training finished — measuring against what was deployed"
  $PY - <<'PYEOF' >> "$LOG" 2>&1
import json, shutil, subprocess, sys
from pathlib import Path

ROOT = Path.cwd()
PY = str(ROOT / ".venv/bin/python")


def overall(weights: Path) -> float:
    """mAP50 on the held-out test split, for one set of weights.

    evaluate.py reads whatever is deployed, so each candidate is swapped in and
    out. One of the candidates *is* the deployed file, and copying a file onto
    itself raises rather than doing nothing — hence the guard.
    """
    live = ROOT / "models/players.pt"
    keep = Path("/tmp/_players_swap.pt")
    shutil.copy(live, keep)
    if Path(weights).resolve() != live.resolve():
        shutil.copy(weights, live)
    try:
        out = subprocess.run(
            [PY, "training/scripts/evaluate.py", "players", "--split", "test"],
            capture_output=True, text=True, cwd=ROOT,
        ).stdout
        blob = out[out.index("{"):out.rindex("}") + 1]
        return float(json.loads(blob)["overall"]["mAP50"])
    finally:
        shutil.copy(keep, live)


before = ROOT / "training/runs/players_deployed_before.pt"
after = ROOT / "models/players.pt"
try:
    old, new = overall(before), overall(after)
except Exception as exc:
    print(f"  could not measure ({exc}) — restoring the previous weights")
    shutil.copy(before, after)
    sys.exit(0)

print(f"  players mAP50 — old {old:.4f}, new {new:.4f}")
if new > old + 0.002:
    print("  new players model is better — keeping it")
else:
    shutil.copy(before, after)
    print("  new players model is not better — restored the previous weights")
PYEOF
fi
say "phase 2 complete"

say "=== both phases done — see $LOG ==="
