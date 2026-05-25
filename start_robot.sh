#!/usr/bin/env bash
set -e

echo "======================================"
echo " Marquinho Bot - Voice + Vision Start "
echo "======================================"
echo

cd "$HOME/robot-project/week3/camera"

if [ -f "$HOME/robot-project/.env" ]; then
  source "$HOME/robot-project/.env"
fi

echo "[1/3] Activating Python environment..."
source "$HOME/robot-project/week3/camera/venv/bin/activate"

echo "[2/3] Checking core files..."
if [ ! -f "$HOME/robot-project/week3/camera/robot_voice_vision.py" ]; then
  echo "ERROR: robot_voice_vision.py not found."
  exit 1
fi

if [ ! -d "$HOME/robot-project/week3/models/vosk-model-small-en-us-0.15" ]; then
  echo "ERROR: Vosk model folder not found."
  exit 1
fi

echo "[3/3] Starting robot..."
echo
python "$HOME/robot-project/week3/camera/robot_voice_vision.py"
