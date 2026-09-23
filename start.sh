#!/usr/bin/env bash
# Start Match Tag and open it in the browser.
#
#   ./start.sh          start it
#   ./start.sh stop     stop it
#   ./start.sh status   see what is running
set -u
cd "$(dirname "$0")"

PORT="${PORT:-5055}"
PY=./.venv/bin/python
LOG=/tmp/matchtag.log

is_running() { pgrep -f "match_tag/app.py" > /dev/null; }

case "${1:-start}" in
  stop)
    pkill -f "match_tag/app.py" && echo "Stopped." || echo "It was not running."
    ;;

  status)
    if is_running; then
      echo "Running on http://localhost:$PORT"
    else
      echo "Not running."
    fi
    if [ -f training/runs/pitch/results.csv ]; then
      awk -F, 'END {printf "Pitch model:   epoch %s of 100\n", $1}' training/runs/pitch/results.csv
    fi
    if [ -f training/runs/players/results.csv ]; then
      awk -F, 'END {printf "Player model:  epoch %s of 100\n", $1}' training/runs/players/results.csv
    fi
    echo "Weights present:"
    for m in players pitch ball; do
      [ -f "models/$m.pt" ] && echo "  $m ✓" || echo "  $m — not trained yet"
    done
    ;;

  *)
    if [ ! -x "$PY" ]; then
      echo "No virtual environment found. Create one first:"
      echo "  python3.10 -m venv .venv"
      echo "  ./.venv/bin/pip install -r match_tag/requirements.txt -r requirements-cv.txt"
      exit 1
    fi
    if is_running; then
      echo "Already running."
    else
      nohup "$PY" match_tag/app.py > "$LOG" 2>&1 &
      # Wait for it to answer rather than guessing at a sleep length.
      for _ in $(seq 1 40); do
        curl -s -o /dev/null --max-time 1 "http://127.0.0.1:$PORT/health" && break
        sleep 0.5
      done
    fi
    if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/health"; then
      echo "Match Tag is running:"
      echo "  http://localhost:$PORT"
      LAN=$(ipconfig getifaddr en0 2>/dev/null || true)
      [ -n "$LAN" ] && echo "  http://$LAN:$PORT   (phone or tablet on the same wifi)"
      echo
      echo "Log: $LOG        Stop it with: ./start.sh stop"
      open "http://localhost:$PORT" 2>/dev/null || true
    else
      echo "It did not start. Last lines of $LOG:"
      tail -20 "$LOG"
      exit 1
    fi
    ;;
esac
