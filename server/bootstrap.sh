#!/usr/bin/env bash
set -euo pipefail

if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
  echo "bootstrap.sh needs bash >= 4 (found $BASH_VERSION)" >&2
  exit 1
fi

bootstrap_main() {

FORCE=0 SKIP_SMOKE=0 CKPT_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --skip-smoke) SKIP_SMOKE=1 ;;
    --checkpoint-only) CKPT_ONLY=1 ;;
    -h|--help) echo "usage: bootstrap.sh [--force] [--skip-smoke] [--checkpoint-only]"; return 0 ;;
    *) echo "unknown argument: $arg" >&2; return 2 ;;
  esac
done

: "${TRACKER_ROOT:?TRACKER_ROOT is not set}"
: "${TRACKER_PYTHON_VERSION:=3.12}"
: "${TRACKER_PYTHON_BIN:=}"
: "${TORCH_VERSION:=2.13.0}"
: "${TORCH_INDEX_URL:=https://download.pytorch.org/whl/cu130}"
: "${TRACKER_PIP_EXTRA_INDEX_URL:=https://pypi.org/simple}"
: "${TRACKER_PIP_PACKAGES:=timm==1.0.25 numpy scipy einops pillow opencv-python-headless huggingface_hub matplotlib}"
: "${TRACKER_XFORMERS_SPEC:=xformers}"
: "${COWTRACKER_REPO:=https://github.com/facebookresearch/cowtracker.git}"
: "${COWTRACKER_COMMIT:=1454f20045d3b514e5b8417907152677f3dba621}"
: "${CHECKPOINT_REPO:=facebook/cowtracker}"
: "${CHECKPOINT_FILE:=cowtracker_model.pth}"
: "${CHECKPOINT_REVISION:=860b85cfb3a63b91780535689c262b3e595b6082}"
: "${CHECKPOINT_URL:=}"
: "${CHECKPOINT_SHA256:=}"
: "${HF_TOKEN:=}"
: "${TRACKER_CUDNN_PREFIX:=}"
: "${TRACKER_MIN_DRIVER:=580}"
: "${TRACKER_GPU:=0}"

ENV_DIR="$TRACKER_ROOT/env"
CODE_DIR="$TRACKER_ROOT/code/cowtracker"
ADAPTER="$TRACKER_ROOT/code/adapter/cowtracker_to_fabsim.py"
CKPT="$TRACKER_ROOT/checkpoints/$CHECKPOINT_FILE"
PY="${TRACKER_PYTHON:-$ENV_DIR/bin/python}"

step() { printf '\n== %s ==\n' "$*"; }
info() { printf '  %s\n' "$*"; }
warn() { printf '  warning: %s\n' "$*" >&2; }
fail() { printf 'error: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "'$1' is required but was not found in PATH"; }

mkdir -p "$TRACKER_ROOT"
cd "$TRACKER_ROOT"

step "preflight"
need curl; need sha256sum
if [ "$CKPT_ONLY" -eq 0 ]; then
  need git
  if [ -n "$TRACKER_PYTHON_BIN" ]; then
    BASE_PY="$TRACKER_PYTHON_BIN"
  elif command -v "python$TRACKER_PYTHON_VERSION" >/dev/null 2>&1; then
    BASE_PY="python$TRACKER_PYTHON_VERSION"
  else
    BASE_PY=python3
  fi
  need "$BASE_PY"
  "$BASE_PY" - "$TRACKER_PYTHON_VERSION" <<'PYCHECK' || fail "$BASE_PY is older than Python $TRACKER_PYTHON_VERSION (set TRACKER_PYTHON_BIN)"
import sys
want = tuple(int(p) for p in sys.argv[1].split("."))
sys.exit(0 if sys.version_info[: len(want)] >= want else 1)
PYCHECK
  "$BASE_PY" -c 'import venv, ensurepip' 2>/dev/null ||
    fail "$BASE_PY cannot create venvs (install python3-venv / python3.12-venv)"
  info "python: $("$BASE_PY" --version 2>&1) ($BASE_PY)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
    info "GPU   : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1), driver $drv"
    [ "${drv%%.*}" -ge "$TRACKER_MIN_DRIVER" ] 2>/dev/null ||
      fail "NVIDIA driver $drv is older than $TRACKER_MIN_DRIVER, which the CUDA build at $TORCH_INDEX_URL needs. Update the driver or pick a TORCH_INDEX_URL for an older CUDA (and lower TRACKER_MIN_DRIVER)."
  else
    fail "nvidia-smi not found: this host has no NVIDIA driver"
  fi
  free_kb="$(df -Pk "$TRACKER_ROOT" | awk 'NR==2 {print $4}')"
  [ "$free_kb" -ge 12000000 ] || warn "less than 12 GB free under $TRACKER_ROOT (torch ~6 GB, checkpoint ~4 GB)"
fi

if [ "$CKPT_ONLY" -eq 0 ]; then
  step "python environment ($ENV_DIR)"
  if [ -x "$ENV_DIR/bin/python" ] && [ "$FORCE" -eq 1 ]; then
    aside="$ENV_DIR.old-$(date -u +%Y%m%dT%H%M%SZ)"
    mv -- "$ENV_DIR" "$aside"
    info "moved the old environment to $aside (delete it yourself when no longer needed)"
  fi
  if [ ! -x "$ENV_DIR/bin/python" ] && [ -z "${TRACKER_PYTHON:-}" ]; then
    "$BASE_PY" -m venv "$ENV_DIR"
    info "created venv"
  else
    info "using existing environment"
  fi
  [ -x "$PY" ] || fail "python of the environment not found: $PY"
  "$PY" -m pip install --quiet --upgrade pip

  step "torch $TORCH_VERSION from $TORCH_INDEX_URL"
  # torch, torchvision and xformers are resolved together so that the xformers
  # build matches the torch build.
  "$PY" -m pip install "torch==$TORCH_VERSION" torchvision "$TRACKER_XFORMERS_SPEC" \
    --index-url "$TORCH_INDEX_URL" --extra-index-url "$TRACKER_PIP_EXTRA_INDEX_URL"
  step "python packages"
  # shellcheck disable=SC2086
  "$PY" -m pip install --index-url "$TRACKER_PIP_EXTRA_INDEX_URL" $TRACKER_PIP_PACKAGES
  # pip may not silently replace the torch we asked for.
  "$PY" - "$TORCH_VERSION" <<'PYCHECK'
import sys, torch
want = sys.argv[1]
if not torch.__version__.startswith(want):
    sys.exit(f"torch {torch.__version__} is installed, expected {want}")
PYCHECK
fi

if [ "$CKPT_ONLY" -eq 0 ]; then
  step "CoWTracker $COWTRACKER_COMMIT"
  if [ ! -e "$CODE_DIR/.git" ]; then
    mkdir -p "$(dirname "$CODE_DIR")"
    git clone --quiet "$COWTRACKER_REPO" "$CODE_DIR"
  fi
  if [ -n "$(git -C "$CODE_DIR" status --porcelain --ignore-submodules 2>/dev/null)" ] && [ "$FORCE" -eq 0 ]; then
    fail "$CODE_DIR has local changes; refusing to check out another revision (use --force)"
  fi
  if ! git -C "$CODE_DIR" cat-file -e "$COWTRACKER_COMMIT^{commit}" 2>/dev/null; then
    git -C "$CODE_DIR" fetch --quiet origin
  fi
  git -C "$CODE_DIR" -c advice.detachedHead=false checkout --quiet --detach "$COWTRACKER_COMMIT"
  # The model imports vggt and Depth-Anything-V2 from git submodules.
  git -C "$CODE_DIR" submodule update --init --recursive --quiet
  info "HEAD $(git -C "$CODE_DIR" rev-parse HEAD)"
  head_now="$(git -C "$CODE_DIR" rev-parse HEAD)"
  case "$head_now" in "$COWTRACKER_COMMIT"*) ;; *) fail "checkout is at $head_now, expected $COWTRACKER_COMMIT" ;; esac
fi

step "checkpoint $CKPT"
mkdir -p "$(dirname "$CKPT")"
url="$CHECKPOINT_URL"
[ -n "$url" ] || url="https://huggingface.co/$CHECKPOINT_REPO/resolve/$CHECKPOINT_REVISION/$CHECKPOINT_FILE"

hash_ckpt() { sha256sum "$1" | cut -d' ' -f1; }
if [ -f "$CKPT" ] && [ "$FORCE" -eq 0 ]; then
  info "already present"
else
  if [ -f "$CKPT" ]; then
    mv -- "$CKPT" "$CKPT.old-$(date -u +%Y%m%dT%H%M%SZ)"
    info "kept the previous file next to it (.old-...)"
  fi
  info "downloading from $url"
  # The .part file lets an interrupted download resume.
  progress=(--silent --show-error)
  [ -t 2 ] && progress=(--progress-bar)
  # The token goes to curl as a config on stdin (-K -), not as an argument,
  # so it never shows up in the process list.
  {
    [ -z "$HF_TOKEN" ] || printf 'header = "Authorization: Bearer %s"\n' "$HF_TOKEN"
  } | curl --fail --location --retry 3 --continue-at - "${progress[@]}" -K - --output "$CKPT.part" "$url"
  mv -- "$CKPT.part" "$CKPT"
  downloaded=1
fi
actual="$(hash_ckpt "$CKPT")"
if [ -n "$CHECKPOINT_SHA256" ]; then
  if [ "$actual" != "$CHECKPOINT_SHA256" ]; then
    if [ "${downloaded:-0}" -eq 1 ]; then
      mv -- "$CKPT" "$CKPT.mismatch"
      fail "downloaded checkpoint has sha256 $actual, expected $CHECKPOINT_SHA256 (kept as $CKPT.mismatch; check CHECKPOINT_REVISION / CHECKPOINT_URL)"
    fi
    fail "checkpoint sha256 mismatch: got $actual, expected $CHECKPOINT_SHA256 (use --force to download again; the old file is kept)"
  fi
  info "sha256 verified: $actual"
else
  warn "CHECKPOINT_SHA256 is empty: not verified. Recording $actual"
fi
printf '%s  %s\n' "$actual" "$CHECKPOINT_FILE" >"$CKPT.sha256"

if [ "$CKPT_ONLY" -eq 1 ]; then
  step "done (checkpoint only)"
  return 0
fi

if [ -n "$TRACKER_CUDNN_PREFIX" ] && [ -d "$TRACKER_CUDNN_PREFIX/nvidia/cudnn/lib" ]; then
  export LD_LIBRARY_PATH="$TRACKER_CUDNN_PREFIX/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [ -n "${TRACKER_LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="$TRACKER_LD_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export CUDA_VISIBLE_DEVICES="$TRACKER_GPU"

gpu_probe() {
  "$PY" - <<'PYPROBE'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
x = torch.randn(1, 3, 16, 16, device="cuda")
w = torch.randn(4, 3, 3, 3, device="cuda")
torch.nn.functional.conv2d(x, w)  # loads cuDNN
print(f"  {torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
      f"CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}")
PYPROBE
}

step "GPU check"
if ! gpu_probe; then
  cat >&2 <<HINT

The GPU check failed. If the message above mentions cuDNN or a missing
libcudnn/libcublas, torch found a different CUDA library than it was built for
(typical on shared or NFS-mounted environments). Fix:
  1. set TRACKER_CUDNN_PREFIX in tracker.env to a directory on this host and
     run setup again: the matching cuDNN wheel is installed there; or
  2. install it yourself:  $PY -m pip install --target <dir> nvidia-cudnn-cu13
     and set TRACKER_LD_LIBRARY_PATH=<dir>/nvidia/cudnn/lib
If torch reports no CUDA device, check the driver (nvidia-smi) and
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES.
HINT
  if [ -n "$TRACKER_CUDNN_PREFIX" ] && [ ! -d "$TRACKER_CUDNN_PREFIX/nvidia/cudnn/lib" ]; then
    info "installing cuDNN into $TRACKER_CUDNN_PREFIX ..."
    "$PY" -m pip install --target "$TRACKER_CUDNN_PREFIX" nvidia-cudnn-cu13
    export LD_LIBRARY_PATH="$TRACKER_CUDNN_PREFIX/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    gpu_probe || fail "GPU check still failing after installing cuDNN"
  else
    fail "GPU check failed"
  fi
fi
PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PY" -c 'import cowtracker, xformers.ops; print("  cowtracker and xformers import fine")' ||
  fail "cowtracker could not be imported (xformers build matching torch? submodules?)"

if [ "$SKIP_SMOKE" -eq 1 ]; then
  step "smoke test skipped"
elif [ ! -f "$ADAPTER" ]; then
  warn "adapter not uploaded to $ADAPTER yet: smoke test skipped (fabsim-track setup uploads it)"
else
  step "smoke test (3 synthetic frames, 4 x 4 points)"
  smoke="$(mktemp -d)"
  trap 'rm -rf "$smoke"' EXIT
  "$PY" - "$smoke" <<'PYSMOKE'
import json, os, sys
import numpy as np
from PIL import Image

root = sys.argv[1]
os.makedirs(os.path.join(root, "images"))
rng = np.random.default_rng(0)
base = (rng.random((96, 128, 3)) * 255).astype(np.uint8)
frames = []
for i in range(3):
    name = f"img_{i:04d}_diffuse.png"
    Image.fromarray(np.roll(base, 2 * i, axis=1)).save(os.path.join(root, "images", name))
    frames.append({"frame_id": i, "image_diffuse": name})
json.dump({"frames": frames}, open(os.path.join(root, "frames.json"), "w"))
PYSMOKE
  "$PY" "$ADAPTER" "$smoke" --grid-cols 4 --grid-rows 4 --checkpoint "$CKPT" \
    --cowtracker-root "$CODE_DIR" --output "$smoke/tracks.json" --verbose
  "$PY" - "$smoke/tracks.json" <<'PYCHECK'
import json, sys
doc = json.load(open(sys.argv[1]))
assert len(doc["frames"]) == 3, "expected 3 frames"
assert all(len(f["points"]) == 16 for f in doc["frames"]), "expected 16 points per frame"
worst = max(doc["provenance"]["reference_errors_px"].values())
assert worst < 2.0, f"reference error {worst} px"
print(f"  smoke test passed (max reference error {worst:.4f} px)")
PYCHECK
fi

step "provenance"
"$PY" -m pip freeze >"$TRACKER_ROOT/env.lock"
"$PY" - "$TRACKER_ROOT/setup.json" "$CODE_DIR" "$CKPT" "$actual" "$COWTRACKER_REPO" <<'PYSETUP'
import datetime, json, platform, socket, subprocess, sys
import torch

path, code, ckpt, sha, repo = sys.argv[1:6]
head = subprocess.run(["git", "-C", code, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
drv = subprocess.run(
    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True
).stdout.split()
doc = {
    "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "host": socket.gethostname(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "nvidia_driver": drv[0] if drv else None,
    "cowtracker_repo": repo,
    "cowtracker_commit": head,
    "checkpoint": ckpt.rsplit("/", 1)[-1],
    "checkpoint_sha256": sha,
}
with open(path, "w") as fh:
    json.dump(doc, fh, indent=2)
    fh.write("\n")
PYSETUP
info "wrote $TRACKER_ROOT/env.lock and $TRACKER_ROOT/setup.json"
step "done"

}

bootstrap_main "$@"
