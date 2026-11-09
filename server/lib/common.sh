# shellcheck shell=bash

CONFIG_KEYS=(
  TRACKER_MODE FABSIM_DATA_ROOT FABSIM_ADAPTER TRACKER_ROOT
  TRACKER_SSH_HOST TRACKER_SSH_KEY TRACKER_SSH_JUMP TRACKER_SSH_OPTS
  TRACKER_GPU TRACKER_LD_LIBRARY_PATH TRACKER_ROLE
  TRACKER_GRID_COLS TRACKER_GRID_ROWS TRACKER_EXTRA_ARGS
  TRACKER_REFERENCE_ERROR_LIMIT TRACKER_DOCKER_IMAGE TRACKER_LOCAL_PYTHON
  TRACKER_PYTHON TRACKER_PYTHON_VERSION TRACKER_PYTHON_BIN
  TORCH_VERSION TORCH_INDEX_URL TRACKER_PIP_EXTRA_INDEX_URL TRACKER_PIP_PACKAGES
  TRACKER_XFORMERS_SPEC COWTRACKER_REPO COWTRACKER_COMMIT
  CHECKPOINT_REPO CHECKPOINT_FILE CHECKPOINT_REVISION CHECKPOINT_URL
  CHECKPOINT_SHA256 HF_TOKEN TRACKER_CUDNN_PREFIX TRACKER_MIN_DRIVER
)

PINNED_COWTRACKER_COMMIT="1454f20045d3b514e5b8417907152677f3dba621"

log()  { printf '%s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit "${DIE_CODE:-1}"; }

# valid_relpath VALUE: "a/b" made of [A-Za-z0-9._-] components, no "." / "..".
valid_relpath() {
  local value="$1" part
  [[ $value =~ ^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ ]] || return 1
  local IFS=/
  for part in $value; do
    [[ $part == . || $part == .. ]] && return 1
  done
  return 0
}

# check_name KIND VALUE REGEX: die with a clear message when VALUE does not match.
check_name() {
  local kind="$1" value="$2" regex="$3"
  [[ $value =~ $regex ]] || die "invalid $kind: '$value' (allowed: $regex)"
}

check_run_id() { check_name "run id" "$1" '^[A-Za-z0-9][A-Za-z0-9._-]*$'; }
check_role()   { check_name "role" "$1" '^[A-Za-z0-9_]+$'; }
check_gpu()    { check_name "GPU list" "$1" '^[0-9]+(,[0-9]+)*$'; }
check_uint()   { check_name "$1" "$2" '^[0-9]+$'; }

# check_abs_path KIND VALUE: absolute path from a conservative character set
# (no spaces or shell metacharacters, so it is safe in rsync/ssh arguments).
check_abs_path() {
  local kind="$1" value="$2"
  [[ $value == /* ]] || die "$kind must be an absolute path (got '$value')"
  [[ $value =~ ^/[A-Za-z0-9._/+@=-]*$ ]] ||
    die "$kind may only contain letters, digits and . _ / + @ = - (got '$value')"
}

# quote_words WORD...: shell-quote each word (for building remote commands).
quote_words() {
  local out="" w
  for w in "$@"; do out+="$(printf '%q' "$w") "; done
  printf '%s' "${out% }"
}

# make_tmp VAR [dir]: create a temp file (or directory) named in VAR, removed
# on exit. (Not usable through $(...), which would lose the cleanup list.)
TMP_PATHS=()
make_tmp() {
  local path
  if [[ ${2:-} == dir ]]; then path="$(mktemp -d)"; else path="$(mktemp)"; fi
  TMP_PATHS+=("$path")
  printf -v "$1" '%s' "$path"
}
cleanup_tmp() {
  if [[ ${#TMP_PATHS[@]} -gt 0 ]]; then rm -rf -- "${TMP_PATHS[@]}"; fi
}

utc_stamp() { date -u +%Y%m%dT%H%M%SZ; }

# find_config_file: print the config file to use (empty when there is none).
# Order: --env / $FABSIM_TRACKER_ENV, server/tracker.env, the user config dir.
find_config_file() {
  local candidate
  for candidate in \
    "${ENV_FILE_ARG:-}" \
    "${FABSIM_TRACKER_ENV:-}" \
    "$SERVER_DIR/tracker.env" \
    "${XDG_CONFIG_HOME:-$HOME/.config}/fabsim/tracker.env"; do
    [[ -n $candidate ]] || continue
    if [[ -f $candidate ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
    # An explicitly requested file must exist.
    if [[ $candidate == "${ENV_FILE_ARG:-}" || $candidate == "${FABSIM_TRACKER_ENV:-}" ]]; then
      die "config file not found: $candidate"
    fi
  done
  return 0
}

# load_config: source the config file without letting it override variables
# that were set explicitly in the environment, then apply defaults.
load_config() {
  local key saved=() file
  for key in "${CONFIG_KEYS[@]}"; do
    if [[ -n ${!key+x} ]]; then
      saved+=("$key=${!key}")
    fi
  done

  CONFIG_FILE="$(find_config_file)"
  if [[ -n $CONFIG_FILE ]]; then
    file="$CONFIG_FILE"
    # shellcheck disable=SC1090
    source "$file"
  fi

  local pair
  for pair in "${saved[@]}"; do
    printf -v "${pair%%=*}" '%s' "${pair#*=}"
  done
  [[ -z ${MODE_ARG:-} ]] || TRACKER_MODE="$MODE_ARG"
  apply_defaults
  for key in "${CONFIG_KEYS[@]}"; do
    [[ $key == HF_TOKEN ]] || export "${key?}"
  done
}

apply_defaults() {
  : "${TRACKER_MODE:=ssh}"
  : "${FABSIM_DATA_ROOT:=}"
  : "${FABSIM_ADAPTER:=$REPO_DIR/src/fabsim/cowtracker_to_fabsim.py}"
  : "${TRACKER_ROOT:=}"
  : "${TRACKER_SSH_HOST:=}"
  : "${TRACKER_SSH_KEY:=}"
  : "${TRACKER_SSH_JUMP:=}"
  : "${TRACKER_SSH_OPTS:=}"
  : "${TRACKER_GPU:=0}"
  : "${TRACKER_LD_LIBRARY_PATH:=}"
  : "${TRACKER_ROLE:=diffuse}"
  : "${TRACKER_GRID_COLS:=20}"
  : "${TRACKER_GRID_ROWS:=16}"
  : "${TRACKER_EXTRA_ARGS:=}"
  : "${TRACKER_REFERENCE_ERROR_LIMIT:=2.0}"
  : "${TRACKER_DOCKER_IMAGE:=fabsim-tracker}"
  : "${TRACKER_LOCAL_PYTHON:=python3}"
  : "${TRACKER_PYTHON:=}"
  : "${TRACKER_PYTHON_VERSION:=3.12}"
  : "${TRACKER_PYTHON_BIN:=}"
  : "${TORCH_VERSION:=2.13.0}"
  : "${TORCH_INDEX_URL:=https://download.pytorch.org/whl/cu130}"
  : "${TRACKER_PIP_EXTRA_INDEX_URL:=https://pypi.org/simple}"
  : "${TRACKER_PIP_PACKAGES:=timm==1.0.25 numpy scipy einops pillow opencv-python-headless huggingface_hub matplotlib}"
  : "${TRACKER_XFORMERS_SPEC:=xformers}"
  : "${COWTRACKER_REPO:=https://github.com/facebookresearch/cowtracker.git}"
  : "${COWTRACKER_COMMIT:=$PINNED_COWTRACKER_COMMIT}"
  : "${CHECKPOINT_REPO:=facebook/cowtracker}"
  : "${CHECKPOINT_FILE:=cowtracker_model.pth}"
  : "${CHECKPOINT_REVISION:=860b85cfb3a63b91780535689c262b3e595b6082}"
  : "${CHECKPOINT_URL:=}"
  : "${CHECKPOINT_SHA256:=5b46c6dac1d9f9f5944371176f5aedec13522db4ea86d998f3445b0fd37ab784}"
  : "${HF_TOKEN:=}"
  : "${TRACKER_CUDNN_PREFIX:=}"
  : "${TRACKER_MIN_DRIVER:=580}"
}

# need_config NAME...: die unless each named setting is non-empty.
need_config() {
  local name
  for name in "$@"; do
    [[ -n ${!name} ]] || die "$name is not set. Copy server/tracker.env.example to server/tracker.env and fill it in (see docs/self_hosting_tracker.md)."
  done
}

# validate_config: strict checks of everything the active mode will use.
validate_config() {
  case "$TRACKER_MODE" in
    ssh|local|docker) ;;
    *) die "TRACKER_MODE must be ssh, local or docker (got '$TRACKER_MODE')" ;;
  esac
  need_config TRACKER_ROOT
  check_abs_path TRACKER_ROOT "$TRACKER_ROOT"
  [[ $TRACKER_ROOT == / ]] && die "TRACKER_ROOT must not be /"
  TRACKER_ROOT="${TRACKER_ROOT%/}"
  check_gpu "$TRACKER_GPU"
  check_role "$TRACKER_ROLE"
  check_uint TRACKER_GRID_COLS "$TRACKER_GRID_COLS"
  check_uint TRACKER_GRID_ROWS "$TRACKER_GRID_ROWS"
  check_name "reference error limit" "$TRACKER_REFERENCE_ERROR_LIMIT" '^[0-9]+(\.[0-9]+)?$'
  check_name "COWTRACKER_COMMIT" "$COWTRACKER_COMMIT" '^[0-9a-f]{7,40}$'
  case "$TRACKER_MODE" in
    ssh)
      need_config TRACKER_SSH_HOST
      check_name "TRACKER_SSH_HOST" "$TRACKER_SSH_HOST" '^[]A-Za-z0-9._@:[-]+$'
      [[ $TRACKER_SSH_HOST != -* ]] || die "TRACKER_SSH_HOST must not start with '-'"
      ;;
    docker)
      check_name "TRACKER_DOCKER_IMAGE" "$TRACKER_DOCKER_IMAGE" '^[A-Za-z0-9][A-Za-z0-9._/:@-]*$'
      ;;
  esac
}

# Directory layout below TRACKER_ROOT (identical for all modes).
root_env()        { printf '%s/env' "$TRACKER_ROOT"; }
root_python()     { printf '%s' "${TRACKER_PYTHON:-$TRACKER_ROOT/env/bin/python}"; }
root_cowtracker() { printf '%s/code/cowtracker' "$TRACKER_ROOT"; }
root_adapter()    { printf '%s/code/adapter/cowtracker_to_fabsim.py' "$TRACKER_ROOT"; }
root_checkpoint() { printf '%s/checkpoints/%s' "$TRACKER_ROOT" "$CHECKPOINT_FILE"; }
