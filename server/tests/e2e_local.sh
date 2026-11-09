#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACK="$HERE/../fabsim-track"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }
expect_fail() { # expect_fail DESCRIPTION CMD...
  local what="$1"; shift
  if "$@" >"$WORK/out.txt" 2>&1; then cat "$WORK/out.txt"; fail "$what: expected failure"; fi
}

DATA="$WORK/data"
for run in warp_loading warp_release bias_loading; do
  mkdir -p "$DATA/set1/$run/images"
  echo '{}' >"$DATA/set1/$run/session.json"
  echo '{}' >"$DATA/set1/$run/force.json"
  python3 - "$DATA/set1/$run" <<'PY'
import json, sys
run = sys.argv[1]
frames = [{"frame_id": i, "image_diffuse": f"img_{i:04d}_diffuse.png", "image_ps_1": f"img_{i:04d}_ps_1.png"} for i in range(3)]
json.dump({"frames": frames}, open(f"{run}/frames.json", "w"))
for f in frames:
    for key in ("image_diffuse", "image_ps_1"):
        open(f"{run}/images/{f[key]}", "wb").write(b"x")
PY
  echo 'unrelated' >"$DATA/set1/$run/images/notes.txt"
done

write_config() { # write_config MODE
  cat >"$WORK/tracker.env" <<CFG
TRACKER_MODE=$1
FABSIM_DATA_ROOT=$DATA
FABSIM_ADAPTER=$HERE/fake_adapter.py
TRACKER_ROOT=$WORK/root
TRACKER_PYTHON=$(command -v python3)
TRACKER_LOCAL_PYTHON=$(command -v python3)
TRACKER_SSH_HOST=fakehost
TRACKER_GRID_COLS=4
TRACKER_GRID_ROWS=3
CFG
}
track() { "$TRACK" --env "$WORK/tracker.env" "$@"; }

for mode in local ssh; do
  echo "=== mode: $mode ==="
  rm -rf "$WORK/root" "$DATA"/set1/*/cowtracker-results "$DATA"/set1/*/tracks.json "$DATA/set1/cowtracker-logs"
  write_config "$mode"
  export PATH="$HERE/fakebin:$PATH"

  # run + results location
  track run set1/warp_loading --run-id r1 --gpu 1 -- --windowed --reference-frame 2 >"$WORK/run.log"
  if [[ $mode == ssh ]]; then
    results="$WORK/root/outputs/set1/warp_loading/r1"
    [[ -f $WORK/root/measurements/set1/warp_loading/images/img_0000_diffuse.png ]] || fail "diffuse image not uploaded"
    [[ ! -e $WORK/root/measurements/set1/warp_loading/images/img_0000_ps_1.png ]] || fail "other role uploaded"
    [[ ! -e $WORK/root/measurements/set1/warp_loading/images/notes.txt ]] || fail "unrelated file uploaded"
  else
    results="$DATA/set1/warp_loading/cowtracker-results/r1"
  fi
  [[ -s $results/tracks.json ]] || fail "no tracks.json in $results"
  grep -qx -- '--windowed' "$results/argv.txt" || fail "adapter args not passed through"
  grep -qx -- '--reference-frame' "$results/argv.txt" || fail "adapter args not passed through"
  [[ -f $WORK/root/code/adapter/cowtracker_to_fabsim.py ]] || fail "adapter not installed"
  [[ -f $WORK/root/code/adapter/cowtracker_to_fabsim.version ]] || fail "version stamp missing"
  grep -q '"cuda_visible_devices": "1"' "$results/tracks.json" || fail "GPU not selected"
  [[ -s $results/run.log ]] || fail "run.log missing"

  # never overwrite
  expect_fail "second run with the same id" track run set1/warp_loading --run-id r1
  track run set1/warp_loading --run-id r1 --force >/dev/null

  # fetch + place + no clobber
  track fetch set1/warp_loading --run-id r1 >"$WORK/fetch.log" 2>&1 || { [[ $mode == local ]] || fail "fetch failed"; }
  [[ -s $DATA/set1/warp_loading/tracks.json ]] || fail "tracks.json not placed"
  echo '{"mine": true}' >"$DATA/set1/warp_loading/tracks.json"
  track fetch set1/warp_loading --run-id r1 >/dev/null 2>&1 || true
  grep -q mine "$DATA/set1/warp_loading/tracks.json" || fail "fetch overwrote an existing tracks.json"
  track fetch set1/warp_loading --run-id r1 --force >/dev/null 2>&1 || true
  ls "$DATA"/set1/warp_loading/tracks.prev-*.json >/dev/null || fail "--force did not keep the previous tracks.json"

  # validate
  track validate set1/warp_loading --run-id r1 >/dev/null || fail "valid run rejected"
  expect_fail "wrong point count" track validate set1/warp_loading --run-id r1 --grid-cols 5
  track validate set1/warp_loading --run-id r1 --any-points >/dev/null || fail "--any-points rejected"
  FAKE_ADAPTER_BAD_REFERENCE=1 track run set1/warp_loading --run-id bad >/dev/null
  track fetch set1/warp_loading --run-id bad --no-place >/dev/null 2>&1 || true
  expect_fail "reference error above limit" track validate set1/warp_loading --run-id bad
  grep -qx -- '--reference-error-limit' "$results/argv.txt" || fail "reference error limit not passed to the adapter"
  grep -q '"source": "set1/warp_loading"' "$results/tracks.json" || fail "run key not passed to the adapter"
  # the limit given to the adapter is the one validate checks
  FAKE_ADAPTER_BAD_REFERENCE=1 track run set1/warp_loading --run-id loose -- --reference-error-limit 9 >/dev/null
  track fetch set1/warp_loading --run-id loose --no-place >/dev/null 2>&1 || true
  track validate set1/warp_loading --run-id loose >/dev/null || fail "run accepted by the adapter was rejected by validate"
  # custom points chosen through TRACKER_EXTRA_ARGS are not held to the grid size
  TRACKER_EXTRA_ARGS="--grid-spacing 10 10" track validate set1/warp_loading --run-id r1 >/dev/null ||
    fail "validate ignored the custom points in TRACKER_EXTRA_ARGS"

  # name validation
  expect_fail "path traversal" track run set1/../x
  expect_fail "shell metacharacters" track run 'set1/warp;loading'
  expect_fail "bad run id" track run set1/warp_loading --run-id 'a b'
  expect_fail "unknown option" track run set1/warp_loading --bogus

  # dataset: one failing run must not stop the others
  expect_fail "dataset with a failing run" env FAKE_ADAPTER_FAIL=bias_loading "$TRACK" --env "$WORK/tracker.env" dataset set1 --run-id d1
  summary="$DATA/set1/cowtracker-logs/d1-summary.json"
  [[ -f $summary ]] || fail "no summary"
  python3 - "$summary" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
status = {r["run"]: r["status"] for r in s["runs"]}
assert status == {"bias_loading": "failed", "warp_loading": "ok", "warp_release": "ok"}, status
assert s["counts"] == {"ok": 2, "failed": 1, "skipped": 0}, s["counts"]
PY
  # a run id used by a single run is not reused by dataset without --force
  track run set1/warp_loading --run-id solo >/dev/null
  rm -f "$DATA/set1/warp_loading/cowtracker-results/solo/tracks.json" "$WORK/root/outputs/set1/warp_loading/solo/tracks.json"
  expect_fail "dataset reusing a foreign run id" "$TRACK" --env "$WORK/tracker.env" dataset set1 warp_loading --run-id solo
  grep -q 'already exists' "$WORK/out.txt" || fail "dataset did not report the existing results directory"
  # resume: finished runs are skipped, the failed one is retried
  track dataset set1 --run-id d1 >"$WORK/resume.log"
  grep -q 'skip' "$WORK/resume.log" || fail "finished runs were not skipped"
  python3 - "$DATA/set1/cowtracker-logs/d1-summary-2.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
assert s["counts"] == {"ok": 1, "failed": 0, "skipped": 2}, s["counts"]
PY
  echo "mode $mode: ok"
done

echo "dry-run:"
track dry-run set1/warp_loading
echo "all fabsim-track end-to-end checks passed"
