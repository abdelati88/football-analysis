#!/usr/bin/env bash
# Keep the Mac running with the lid closed until training finishes, then put
# every setting back the way it was.
#
#   sudo ./keep_awake.sh
#
# Leave the machine on a desk, on the charger. Not in a bag: this MacBook Air
# has no fan, and the keyboard deck is one of the surfaces it sheds heat
# through. Closing the lid on a hard surface is fine; wrapping it in fabric is
# not.
set -u
cd "$(dirname "$0")"

if [ "$(id -u)" -ne 0 ]; then
  echo "Run it with sudo — changing the lid-sleep setting needs it:"
  echo "  sudo ./keep_awake.sh"
  exit 1
fi

restore() {
  echo
  echo "[$(date '+%H:%M:%S')] restoring normal sleep behaviour"
  pmset -a disablesleep 0
  pkill -x caffeinate 2>/dev/null
  echo "Done. The Mac will sleep normally again."
}
# Restore on Ctrl+C or a terminal close, not only on a clean finish.
trap restore EXIT INT TERM

if ! pgrep -f "train.py" > /dev/null; then
  echo "No training is running. Nothing to stay awake for."
  exit 0
fi

echo "Keeping the Mac awake with the lid closed."
pmset -a disablesleep 1
caffeinate -dimsu &

power=$(pmset -g batt | grep -o "'.*'" | tr -d "'")
echo "  power source: $power"
[ "$power" != "AC Power" ] && echo "  ⚠️  Not on the charger. Plug it in."
echo "  You can close the lid now. Leave it on a desk, not in a bag."
echo

while pgrep -f "train.py" > /dev/null; do
  epoch=$(awk -F, 'END {print $1}' training/runs/pitch/results.csv 2>/dev/null)
  printf "\r  [%s] training… epoch %s of 100   " "$(date '+%H:%M:%S')" "${epoch:-?}"
  sleep 60
done

echo
echo "[$(date '+%H:%M:%S')] training finished."
if [ -f training/runs/pitch/weights/best.pt ]; then
  cp training/runs/pitch/weights/best.pt models/pitch.pt
  echo "Installed the best checkpoint into models/pitch.pt"
fi
