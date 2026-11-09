# shellcheck shell=bash

BOOTSTRAP_KEYS=(
  TRACKER_ROOT TRACKER_PYTHON_VERSION TRACKER_PYTHON_BIN TORCH_VERSION
  TORCH_INDEX_URL TRACKER_PIP_EXTRA_INDEX_URL TRACKER_PIP_PACKAGES
  TRACKER_XFORMERS_SPEC COWTRACKER_REPO COWTRACKER_COMMIT CHECKPOINT_REPO
  CHECKPOINT_FILE CHECKPOINT_REVISION CHECKPOINT_URL CHECKPOINT_SHA256
  HF_TOKEN TRACKER_CUDNN_PREFIX TRACKER_LD_LIBRARY_PATH TRACKER_PYTHON
  TRACKER_MIN_DRIVER TRACKER_GPU
)

bootstrap_script() {
  local key
  for key in "${BOOTSTRAP_KEYS[@]}"; do
    printf 'export %s=%q\n' "$key" "${!key}"
  done

  cat -- "$SERVER_DIR/bootstrap.sh"
}

cmd_setup() {
  parse_opts --force --skip-smoke --checkpoint-only -- "$@"
  local args=()
  if [[ $OPT_FORCE -eq 1 ]]; then args+=(--force); fi
  if [[ $OPT_SKIP_SMOKE -eq 1 ]]; then args+=(--skip-smoke); fi
  if [[ $OPT_CKPT_ONLY -eq 1 ]]; then args+=(--checkpoint-only); fi

  log "== setup: mode $TRACKER_MODE, target $(t_where), root $TRACKER_ROOT =="
  require_target
  case "$TRACKER_MODE" in
    docker)
      command -v docker >/dev/null || die "docker not found in PATH"
      log "== docker build: $TRACKER_DOCKER_IMAGE =="
      docker build -f "$SERVER_DIR/Dockerfile" \
        --build-arg "COWTRACKER_REPO=$COWTRACKER_REPO" \
        --build-arg "COWTRACKER_COMMIT=$COWTRACKER_COMMIT" \
        --build-arg "TORCH_VERSION=$TORCH_VERSION" \
        --build-arg "TORCH_INDEX_URL=$TORCH_INDEX_URL" \
        --build-arg "TRACKER_XFORMERS_SPEC=$TRACKER_XFORMERS_SPEC" \
        --build-arg "TRACKER_PIP_PACKAGES=$TRACKER_PIP_PACKAGES" \
        -t "$TRACKER_DOCKER_IMAGE" "$REPO_DIR" || die "docker build failed"
      # The image contains no weights: fetch the checkpoint to be mounted.
      args+=(--checkpoint-only)
      ;;
    *)
      upload_adapter
      ;;
  esac
  bootstrap_script | t_sh "${args[@]}" || die "bootstrap failed on $(t_where) (see the messages above)"
  log "== setup finished; check with: server/fabsim-track status =="
}

# status_script: a script for the target that prints "  ok|warn|FAIL ..." lines.
status_script() {
  printf 'ROOT=%q PY=%q CKPT=%q COMMIT=%q SHA=%q MODE=%q MIN_DRIVER=%q DEEP=%q\n' \
    "$TRACKER_ROOT" "$(root_python)" "$(root_checkpoint)" "$COWTRACKER_COMMIT" \
    "$CHECKPOINT_SHA256" "$TRACKER_MODE" "$TRACKER_MIN_DRIVER" "$OPT_DEEP"
  ld_exports
  cat <<'SCRIPT'
ok()   { echo "  ok    $*"; }
warn() { echo "  warn  $*"; }
bad()  { echo "  FAIL  $*"; }

echo "target : $(hostname) ($(uname -sr))"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,driver_version --format=csv,noheader |
    sed 's/^/  gpu   /'
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
  if [ "${drv%%.*}" -ge "$MIN_DRIVER" ] 2>/dev/null; then
    ok "NVIDIA driver $drv (>= $MIN_DRIVER)"
  else
    bad "NVIDIA driver '$drv' is older than $MIN_DRIVER (needed for the configured CUDA build)"
  fi
else
  bad "nvidia-smi not found: no NVIDIA driver on this host"
fi

if [ "$MODE" != docker ]; then
  if [ -x "$PY" ]; then
    if info=$("$PY" -c 'import sys, torch; print(sys.version.split()[0], torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "cudnn", torch.backends.cudnn.version())' 2>&1); then
      case "$info" in
        *"available True"*) ok "python/torch: $info" ;;
        *) bad "torch cannot use CUDA: $info" ;;
      esac
    else
      bad "torch import failed: $(echo "$info" | tail -n1)"
    fi
  else
    bad "python environment missing: $PY (run: server/fabsim-track setup)"
  fi
  checkout="$ROOT/code/cowtracker"
  if [ -d "$checkout/.git" ] || [ -f "$checkout/.git" ]; then
    head=$(git -C "$checkout" rev-parse HEAD 2>/dev/null || true)
    if [ "$head" = "$COMMIT" ] || { [ ${#COMMIT} -lt 40 ] && [ "${head#"$COMMIT"}" != "$head" ]; }; then
      ok "CoWTracker checkout at ${head:0:12}"
    else
      bad "CoWTracker checkout is at ${head:-unknown}, expected $COMMIT"
    fi
    if [ -z "$(ls -A "$checkout/cowtracker/thirdparty/vggt" 2>/dev/null)" ]; then
      bad "submodule cowtracker/thirdparty/vggt is empty (git submodule update --init --recursive)"
    fi
  else
    bad "CoWTracker checkout missing: $checkout"
  fi
fi

if [ -f "$CKPT" ]; then
  size=$(du -h "$CKPT" | cut -f1)
  rec=$(cut -d' ' -f1 "$CKPT.sha256" 2>/dev/null || true)
  if [ -n "$SHA" ] && [ "$rec" = "$SHA" ]; then
    ok "checkpoint $size, recorded sha256 matches the expected one"
  elif [ -n "$rec" ]; then
    warn "checkpoint $size, recorded sha256 ${rec:0:16}... differs from CHECKPOINT_SHA256"
  else
    warn "checkpoint $size, no recorded sha256 (re-run setup to record it)"
  fi
  if [ "$DEEP" = 1 ]; then
    now=$(sha256sum "$CKPT" | cut -d' ' -f1)
    if [ "$now" = "${SHA:-$rec}" ]; then ok "checkpoint re-hashed: matches"; else bad "checkpoint re-hashed: $now does not match"; fi
  fi
else
  bad "checkpoint missing: $CKPT"
fi

stamp="$ROOT/code/adapter/cowtracker_to_fabsim.version"
if [ -f "$stamp" ]; then ok "adapter uploaded: $(tr '\n' ' ' < "$stamp")"; else warn "adapter not uploaded yet (done by run/setup)"; fi
[ -f "$ROOT/setup.json" ] && ok "setup.json: $(tr -d '\n ' < "$ROOT/setup.json" | cut -c1-120)"
echo "  disk  $(df -h "$ROOT" 2>/dev/null | awk 'NR==2 {print $4 " free of " $2 " (" $5 " used)"}')"
runs=$(find "$ROOT/outputs" -mindepth 3 -maxdepth 3 -type d -printf '%T@ %P\n' 2>/dev/null | sort -rn | head -n5 | cut -d' ' -f2-)
if [ -n "$runs" ]; then echo "  last runs:"; echo "$runs" | sed 's/^/    /'; fi
SCRIPT
}

cmd_status() {
  parse_opts --deep -- "$@"
  log "mode    : $TRACKER_MODE"
  log "config  : ${CONFIG_FILE:-<none, built-in defaults>}"
  log "root    : $TRACKER_ROOT"
  if target_is_remote; then log "host    : $TRACKER_SSH_HOST"; fi
  log "data    : ${FABSIM_DATA_ROOT:-<not set>}"
  adapter_info
  log "adapter : $FABSIM_ADAPTER (version ${ADAPTER_VERSION:-unknown}, commit ${ADAPTER_COMMIT:-unknown})"
  local tool missing=0 tools=("$TRACKER_LOCAL_PYTHON")
  if target_is_remote; then tools+=(ssh rsync); fi
  for tool in "${tools[@]}"; do
    command -v "$tool" >/dev/null 2>&1 || { log "  FAIL  local tool not found: $tool"; missing=1; }
  done
  if [[ $TRACKER_MODE == docker ]]; then
    if ! command -v docker >/dev/null 2>&1; then
      log "  FAIL  docker not found"; missing=1
    elif docker image inspect "$TRACKER_DOCKER_IMAGE" >/dev/null 2>&1; then
      log "  ok    docker image $TRACKER_DOCKER_IMAGE present"
    else
      log "  FAIL  docker image $TRACKER_DOCKER_IMAGE missing (run: server/fabsim-track setup)"; missing=1
    fi
  fi
  require_target
  local report
  make_tmp report
  status_script | t_sh | tee "$report"
  if [[ $missing -ne 0 ]] || grep -q '^  FAIL' "$report"; then
    die "status: problems found (see FAIL lines above)"
  fi
  log "status: all checks passed"
}
