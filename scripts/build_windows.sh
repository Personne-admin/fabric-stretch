#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_WIN="${PY_WIN:-C:\\Python312\\python.exe}"
NAME="fabsim-rig"
WORK="$ROOT/build/winbuild"
DIST="$ROOT/dist"
LOG="$WORK/pyinstaller.log"
RUN=0
[[ "${1:-}" == "--run" ]] && RUN=1

export WINEDEBUG=-all

command -v wine >/dev/null || { echo "error: wine not found" >&2; exit 1; }

if ! wine "$PY_WIN" -c "import PyInstaller, serial, PIL, PIL.ImageTk, numpy, cv2, tkinter" >/dev/null 2>&1; then
    echo "error: $PY_WIN lacks PyInstaller, pyserial, pillow (with ImageTk), numpy, opencv or tkinter;" >&2
    echo "       see the setup notes at the top of this script" >&2
    exit 1
fi

if ! wine "$PY_WIN" -c "import pyueye" >/dev/null 2>&1; then
    echo "warning: pyueye is not installed in $PY_WIN; the exe will not be able to open the uEye camera" >&2
fi

mkdir -p "$WORK" "$DIST"

WIN_SRC="$(winepath -w "$ROOT/src")"
WIN_ENTRY="$(winepath -w "$ROOT/src/fabsim/rig/__main__.py")"
WIN_DEFAULTS="$(winepath -w "$ROOT/src/fabsim/rig/defaults")"
WIN_WORK="$(winepath -w "$WORK")"
WIN_DIST="$(winepath -w "$DIST")"

echo "Building $NAME.exe (log: $LOG)"
set +e

wine "$PY_WIN" -m PyInstaller --noconfirm --onefile --windowed --name "$NAME" \
    --paths "$WIN_SRC" \
    --collect-submodules fabsim.rig \
    --hidden-import fabsim.color_calibration \
    --hidden-import fabsim.photometric \
    --hidden-import pyueye.ueye \
    --add-data "$WIN_DEFAULTS;fabsim/rig/defaults" \
    --exclude-module torch \
    --workpath "$WIN_WORK\\work" --specpath "$WIN_WORK" --distpath "$WIN_DIST" \
    "$WIN_ENTRY" > "$LOG" 2>&1
status=$?
set -e

if [[ $status -ne 0 || ! -f "$DIST/$NAME.exe" ]]; then
    echo "error: build failed (exit $status), last lines of $LOG:" >&2
    tail -n 20 "$LOG" >&2
    exit 1
fi

echo "Built $DIST/$NAME.exe ($(du -h "$DIST/$NAME.exe" | cut -f1))"

if [[ $RUN -eq 1 ]]; then
    echo "Launching $NAME.exe under Wine (detached, log: $WORK/run.log)"
    cd "$WORK"
    nohup wine "$DIST/$NAME.exe" > "$WORK/run.log" 2>&1 &
    echo "pid $!"
fi
