#!/usr/bin/env bash
# WALP - Linux launcher, the counterpart of play_walp2.bat.
#
#     bash play_walp2.sh [--windowed] [--camera 1] ...

cd "$(dirname "$0")" || exit 1

# A machine can have more than one Python. Only the one with the libraries
# installed can run the game, so pick whichever actually imports them instead
# of trusting whatever "python3" happens to point at. A venv in this folder
# comes first: recent Ubuntu and Debian refuse a system-wide pip install, so
# that is where the libraries usually are.
PYEXE=""
for c in .venv/bin/python venv/bin/python python3.11 python3.12 python3.10 python3 python; do
    if command -v "$c" >/dev/null 2>&1 &&
        "$c" -c "import pygame, cv2, mediapipe, numpy" >/dev/null 2>&1; then
        PYEXE="$c"
        break
    fi
done

if [ -z "$PYEXE" ]; then
    echo
    echo "---------------------------------------------------------------"
    echo " No Python found with the required libraries installed."
    echo
    echo " Install them with:"
    echo "    python3 -m pip install pygame opencv-python mediapipe numpy"
    echo
    echo " Note: mediapipe does not support Python 3.13 yet - use 3.9-3.12."
    echo "---------------------------------------------------------------"
    echo
    exit 1
fi

echo "Using $PYEXE"
exec "$PYEXE" walp2.py "$@"
