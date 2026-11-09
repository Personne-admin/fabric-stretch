# shellcheck shell=bash

parse_opts() {
  local allowed=() arg
  while [[ $# -gt 0 && $1 != -- ]]; do allowed+=("$1"); shift; done
  shift
  OPT_ROLE="$TRACKER_ROLE"; OPT_COLS="$TRACKER_GRID_COLS"; OPT_ROWS="$TRACKER_GRID_ROWS"
  OPT_GPU="$TRACKER_GPU"; OPT_RUN_ID=""; OPT_FORCE=0; OPT_NO_PLACE=0
  OPT_TRACKS=""; OPT_EXPECT=""; OPT_ANY_POINTS=0
  OPT_SKIP_SMOKE=0; OPT_CKPT_ONLY=0; OPT_DEEP=0; OPT_REFETCH=0
  POSITIONAL=(); ADAPTER_EXTRA=()
  while [[ $# -gt 0 ]]; do
    arg="$1"
    if [[ $arg == -- ]]; then
      shift
      ADAPTER_EXTRA=("$@")
      break
    fi
    if [[ $arg == -* ]]; then
      local ok=0 a
      for a in "${allowed[@]}"; do [[ $a == "$arg" ]] && ok=1; done
      [[ $ok -eq 1 ]] || die "unknown option for '$COMMAND': $arg (see: fabsim-track help $COMMAND)"
    fi
    case "$arg" in
      --role)         OPT_ROLE="${2:?--role needs a value}"; shift ;;
      --grid-cols)    OPT_COLS="${2:?--grid-cols needs a value}"; shift ;;
      --grid-rows)    OPT_ROWS="${2:?--grid-rows needs a value}"; shift ;;
      --gpu)          OPT_GPU="${2:?--gpu needs a value}"; shift ;;
      --run-id)       OPT_RUN_ID="${2:?--run-id needs a value}"; shift ;;
      --tracks)       OPT_TRACKS="${2:?--tracks needs a value}"; shift ;;
      --expect-points) OPT_EXPECT="${2:?--expect-points needs a value}"; shift ;;
      --any-points)   OPT_ANY_POINTS=1 ;;
      --force)        OPT_FORCE=1 ;;
      --no-place)     OPT_NO_PLACE=1 ;;
      --skip-smoke)   OPT_SKIP_SMOKE=1 ;;
      --checkpoint-only) OPT_CKPT_ONLY=1 ;;
      --deep)         OPT_DEEP=1 ;;
      --refetch)      OPT_REFETCH=1 ;;
      *)              POSITIONAL+=("$arg") ;;
    esac
    shift
  done
  check_role "$OPT_ROLE"
  check_uint "--grid-cols" "$OPT_COLS"
  check_uint "--grid-rows" "$OPT_ROWS"
  check_gpu "$OPT_GPU"
  if [[ -n $OPT_RUN_ID ]]; then check_run_id "$OPT_RUN_ID"; fi
  if [[ -n $OPT_EXPECT ]]; then check_uint "--expect-points" "$OPT_EXPECT"; fi
  return 0
}

# resolve_run ARG: set RUN_DIR (absolute, contains frames.json) and RUN_KEY
# (the "<dataset>/<run>" name used below TRACKER_ROOT on the target).
resolve_run() {
  local arg="${1%/}" parent
  [[ -n $arg ]] || die "missing run argument (<dataset>/<run> or an absolute run directory)"
  if [[ $arg == /* ]]; then
    RUN_DIR="$(cd -- "$arg" 2>/dev/null && pwd)" || die "run directory not found: $arg"
    parent="$(basename -- "$(dirname -- "$RUN_DIR")")"
    RUN_KEY="$parent/$(basename -- "$RUN_DIR")"
  else
    valid_relpath "$arg" || die "invalid run name '$arg' (expected <dataset>/<run> using letters, digits and . _ -)"
    need_config FABSIM_DATA_ROOT
    RUN_DIR="$FABSIM_DATA_ROOT/$arg"
    RUN_KEY="$arg"
    [[ -d $RUN_DIR ]] || die "run directory not found: $RUN_DIR (FABSIM_DATA_ROOT=$FABSIM_DATA_ROOT)"
    RUN_DIR="$(cd -- "$RUN_DIR" && pwd)"
  fi
  valid_relpath "$RUN_KEY" || die "run path components may only contain letters, digits and . _ - (got '$RUN_KEY')"
  [[ -f $RUN_DIR/frames.json ]] || die "no frames.json in $RUN_DIR"
}

# require_target: make sure the target answers (ssh mode: reachable).
require_target() {
  if target_is_remote; then
    t_sh <<<'true' || die "cannot reach $TRACKER_SSH_HOST over ssh (VPN up? jump host? see docs/self_hosting_tracker.md#troubleshooting)"
  fi
}

# split_extra_args: put TRACKER_EXTRA_ARGS (whitespace separated) in CONFIG_EXTRA.
split_extra_args() {
  CONFIG_EXTRA=()
  if [[ -n $TRACKER_EXTRA_ARGS ]]; then
    read -r -a CONFIG_EXTRA <<<"$TRACKER_EXTRA_ARGS"
  fi
}

build_adapter_argv() {
  local run_path="$1" ckpt="$2" cow="$3" out="$4" a custom=0
  split_extra_args
  for a in "${CONFIG_EXTRA[@]}" "${ADAPTER_EXTRA[@]}"; do
    case "$a" in --points|--points=*|--grid-spacing|--grid-cols|--grid-rows) custom=1 ;; esac
  done

  ADAPTER_ARGV=("$run_path" --role "$OPT_ROLE" --reference-error-limit "$TRACKER_REFERENCE_ERROR_LIMIT")
  EXPECT_POINTS=""
  if [[ $custom -eq 0 ]]; then
    ADAPTER_ARGV+=(--grid-cols "$OPT_COLS" --grid-rows "$OPT_ROWS")
    EXPECT_POINTS=$((OPT_COLS * OPT_ROWS))
  fi

  if [[ -n $out ]]; then
    ADAPTER_ARGV+=(--checkpoint "$ckpt" --cowtracker-root "$cow"
      --output "$out/tracks.json" --csv-output "$out/tracks.csv" --diagnostics-dir "$out/diagnostics")
  fi

  ADAPTER_ARGV+=("${CONFIG_EXTRA[@]}" "${ADAPTER_EXTRA[@]}")
}

# result_dir RUN_ID: where the results of one run live on the target.
result_dir() {
  if target_is_remote; then
    printf '%s/outputs/%s/%s' "$TRACKER_ROOT" "$RUN_KEY" "$1"
  else
    printf '%s/cowtracker-results/%s' "$RUN_DIR" "$1"
  fi
}

# local_results RUN_ID: the local archive directory of one run.
local_results() { printf '%s/cowtracker-results/%s' "$RUN_DIR" "$1"; }

cmd_dry_run() {
  parse_opts --role --grid-cols --grid-rows -- "$@"
  resolve_run "${POSITIONAL[0]:-}"
  adapter_info
  build_adapter_argv "$RUN_DIR" "" "" ""
  log "== dry run (local, no GPU, no torch): $RUN_KEY =="
  local output rc=0
  output="$("$TRACKER_LOCAL_PYTHON" "$FABSIM_ADAPTER" "${ADAPTER_ARGV[@]}" --dry-run 2>&1)" || rc=$?
  printf '%s\n' "$output"
  if [[ $rc -ne 0 && $output == *ModuleNotFoundError* ]]; then
    warn "dry-run runs the adapter here and needs numpy and Pillow: run 'pip install -e .' in the repository and set TRACKER_LOCAL_PYTHON=<repo>/.venv/bin/python"
  fi

  return "$rc"
}

ld_exports() {
  # shellcheck disable=SC2016  # the ${...} must reach the target unexpanded
  {
    if [[ -n $TRACKER_LD_LIBRARY_PATH ]]; then
      printf 'export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' "$TRACKER_LD_LIBRARY_PATH"
    fi
    if [[ -n $TRACKER_CUDNN_PREFIX ]]; then
      printf 'export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' "$TRACKER_CUDNN_PREFIX/nvidia/cudnn/lib"
    fi
  }
}

# remote_script: print the script that runs the adapter (ssh and local mode).
# $1 = run path on the target, $2 = result dir.
remote_script() {
  local run_path="$1" out="$2" py ckpt
  py="$(root_python)"
  ckpt="$(root_checkpoint)"
  build_adapter_argv "$run_path" "$ckpt" "$(root_cowtracker)" "$out"
  printf 'set -euo pipefail\n'
  ld_exports
  printf 'export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=%q\n' "$OPT_GPU"
  printf 'export FABSIM_RUN_KEY=%q\n' "$RUN_KEY"
  if [[ -n ${ADAPTER_COMMIT:-} ]]; then
    printf 'export FABSIM_ADAPTER_COMMIT=%q\n' "$ADAPTER_COMMIT"
  fi

  # shellcheck disable=SC2016
  printf 'ckpt=%q\nif [ -r "$ckpt.sha256" ]; then COWTRACKER_CHECKPOINT_SHA256=$(cut -d" " -f1 < "$ckpt.sha256"); export COWTRACKER_CHECKPOINT_SHA256; fi\n' "$ckpt"
  printf 'if [ ! -x %q ]; then echo "tracker environment missing under %s: run server/fabsim-track setup" >&2; exit 4; fi\n' "$py" "$TRACKER_ROOT"
  printf 'mkdir -p %q\n' "$out/diagnostics"
  printf '%s %q %s 2>&1 | tee -a %q\n' "$(quote_words "$py")" "$(root_adapter)" "$(quote_words "${ADAPTER_ARGV[@]}")" "$out/run.log"
}

# docker_run OUT: run the adapter inside the tracker image.
docker_run() {
  local out="$1"
  build_adapter_argv /data/run "/ckpt/$CHECKPOINT_FILE" /opt/cowtracker /data/out
  local env_args=(-e HOME=/tmp -e PYTHONUNBUFFERED=1 -e "FABSIM_RUN_KEY=$RUN_KEY")
  if [[ -n ${ADAPTER_COMMIT:-} ]]; then env_args+=(-e "FABSIM_ADAPTER_COMMIT=$ADAPTER_COMMIT"); fi
  local sha_file
  sha_file="$(root_checkpoint).sha256"
  if [[ -r $sha_file ]]; then
    env_args+=(-e "COWTRACKER_CHECKPOINT_SHA256=$(cut -d' ' -f1 <"$sha_file")")
  fi

  mkdir -p -- "$out/diagnostics"
  docker run --rm --gpus "\"device=$OPT_GPU\"" --user "$(id -u):$(id -g)" "${env_args[@]}" \
    -v "$RUN_DIR:/data/run:ro" -v "$out:/data/out" \
    -v "$TRACKER_ROOT/checkpoints:/ckpt:ro" \
    -v "$FABSIM_ADAPTER:/opt/fabsim/cowtracker_to_fabsim.py:ro" \
    "$TRACKER_DOCKER_IMAGE" "${ADAPTER_ARGV[@]}" 2>&1 | tee -a "$out/run.log"
}

cmd_run() {
  parse_opts --role --grid-cols --grid-rows --gpu --run-id --force -- "$@"
  resolve_run "${POSITIONAL[0]:-}"
  local run_id="${OPT_RUN_ID:-$(utc_stamp)}" out list n
  check_run_id "$run_id"
  out="$(result_dir "$run_id")"
  adapter_info
  require_target

  make_tmp list
  n="$("$TRACKER_LOCAL_PYTHON" "$LIB_DIR/run_info.py" inputs --run "$RUN_DIR" --role "$OPT_ROLE" --out "$list")" ||
    die "run $RUN_KEY has no usable '$OPT_ROLE' images"

  log "== preflight =="
  log "  mode        : $TRACKER_MODE ($(t_where))"
  log "  run         : $RUN_DIR"
  log "  role/images : $OPT_ROLE / $n"
  log "  GPU         : $OPT_GPU"
  log "  run id      : $run_id"
  log "  results     : $out"

  # Never overwrite a previous run unless asked to.
  if t_exists "$out"; then
    if [[ $OPT_FORCE -eq 1 ]]; then
      warn "results directory exists, --force given: reusing $out"
    else
      DIE_CODE=3 die "results directory already exists: $out (use another --run-id, or --force to reuse it)"
    fi
  fi

  if [[ $TRACKER_MODE != docker ]]; then upload_adapter; fi

  local run_path="$RUN_DIR"
  if target_is_remote; then
    run_path="$TRACKER_ROOT/measurements/$RUN_KEY"
    log "== upload inputs ($n $OPT_ROLE images + session files) =="
    t_push_files "$RUN_DIR" "$list" "$run_path" || die "could not upload the inputs"
  fi

  log "== tracking ($RUN_KEY, run id $run_id) =="
  if [[ $TRACKER_MODE == docker ]]; then
    docker_run "$out" || die "tracking failed (log: $out/run.log)"
  else
    # The script creates the results directory itself, after it has checked
    # the environment, so a missing environment leaves nothing behind.
    remote_script "$run_path" "$out" | t_sh ||
      die "tracking failed (log: $(t_where):$out/run.log; to retry under the same run id add --force)"
  fi

  log "== done =="
  log "  results : $(t_where):$out"
  log "  next    : $SELF fetch $RUN_KEY --run-id $run_id"
}

# place_tracks SRC FORCE: copy SRC to <run>/tracks.json without
# clobbering an existing, different file.
place_tracks() {
  local src="$1" force="$2" dest="$RUN_DIR/tracks.json" prev
  if [[ ! -e $dest ]]; then
    cp -- "$src" "$dest"
    log "  placed  : $dest"
  elif cmp -s "$src" "$dest"; then
    log "  placed  : $dest (already identical)"
  elif [[ $force -eq 1 ]]; then
    prev="$RUN_DIR/tracks.prev-$(utc_stamp).json"
    mv -- "$dest" "$prev"
    cp -- "$src" "$dest"
    log "  placed  : $dest (previous file kept as $(basename "$prev"))"
  else
    warn "$dest already exists and differs; not replaced (use --force, or load $src in the viewer)"
  fi
}

cmd_fetch() {
  parse_opts --run-id --force --refetch --no-place -- "$@"
  resolve_run "${POSITIONAL[0]:-}"
  [[ -n $OPT_RUN_ID ]] || die "fetch needs --run-id"
  local dest remote
  dest="$(local_results "$OPT_RUN_ID")"
  if target_is_remote; then
    require_target
    remote="$(result_dir "$OPT_RUN_ID")"
    if [[ -e $dest && $OPT_FORCE -ne 1 && $OPT_REFETCH -ne 1 ]]; then
      DIE_CODE=3 die "local archive already exists: $dest (pass --refetch to download again)"
    fi
    t_exists "$remote" || die "no such run on $(t_where): $remote"
    log "== fetch $RUN_KEY ($OPT_RUN_ID) =="
    log "  from : $(t_where):$remote"
    log "  to   : $dest"
    t_pull_dir "$remote" "$dest" || die "download failed"
  else
    log "== fetch $RUN_KEY ($OPT_RUN_ID): results are local =="
  fi
  [[ -s $dest/tracks.json ]] || die "tracks.json missing or empty in $dest"
  log "  files   : $(find "$dest" -type f | wc -l) ($(du -sh "$dest" | cut -f1))"
  if [[ $OPT_NO_PLACE -eq 0 ]]; then place_tracks "$dest/tracks.json" "$OPT_FORCE"; fi
  log "  open in the viewer : cargo run --release -p fabric-viewer -- \"$RUN_DIR\""
}

cmd_validate() {
  parse_opts --run-id --tracks --role --grid-cols --grid-rows --expect-points --any-points -- "$@"
  resolve_run "${POSITIONAL[0]:-}"
  local tracks expect=() rc=0
  if [[ -n $OPT_TRACKS ]]; then
    tracks="$OPT_TRACKS"
  elif [[ -n $OPT_RUN_ID ]]; then
    tracks="$(local_results "$OPT_RUN_ID")/tracks.json"
  else
    tracks="$RUN_DIR/tracks.json"
  fi
  # The point count follows the grid unless TRACKER_EXTRA_ARGS (or "-- args"
  # given here) choose the points themselves.
  build_adapter_argv "" "" "" ""
  if [[ $OPT_ANY_POINTS -eq 0 && -n ${OPT_EXPECT:-$EXPECT_POINTS} ]]; then
    expect=(--expect-points "${OPT_EXPECT:-$EXPECT_POINTS}")
  fi
  "$TRACKER_LOCAL_PYTHON" "$LIB_DIR/run_info.py" validate --run "$RUN_DIR" --tracks "$tracks" \
    --role "$OPT_ROLE" "${expect[@]}" || rc=$?
  if [[ $rc -ne 0 && ${#expect[@]} -gt 0 ]]; then
    warn "if this run used --points, --grid-spacing or another grid, repeat with --any-points (or --expect-points N)"
  fi
  return "$rc"
}

# discover_runs DATASET_DIR: <orientation>_<loading|release> runs with frames.json.
discover_runs() {
  local d
  for d in "$1"/*_loading "$1"/*_release; do
    [[ -f $d/frames.json ]] && basename -- "$d"
  done | sort
}

cmd_dataset() {
  parse_opts --role --grid-cols --grid-rows --gpu --run-id --force --no-place -- "$@"
  local ds="${POSITIONAL[0]:-}" ds_dir ds_key run
  [[ -n $ds ]] || die "dataset needs a name: server/fabsim-track dataset <dataset> [run ...]"
  if [[ $ds == /* ]]; then
    ds_dir="$(cd -- "${ds%/}" 2>/dev/null && pwd)" || die "dataset not found: $ds"
    ds_key="$(basename -- "$ds_dir")"
  else
    valid_relpath "${ds%/}" || die "invalid dataset name '$ds'"
    need_config FABSIM_DATA_ROOT
    ds_key="${ds%/}"
    ds_dir="$FABSIM_DATA_ROOT/$ds_key"
    [[ -d $ds_dir ]] || die "dataset not found: $ds_dir"
  fi

  local runs=("${POSITIONAL[@]:1}")
  if [[ ${#runs[@]} -eq 0 ]]; then
    mapfile -t runs < <(discover_runs "$ds_dir")
  fi
  [[ ${#runs[@]} -gt 0 ]] || die "no <orientation>_loading/_release runs with frames.json under $ds_dir"
  for run in "${runs[@]}"; do
    valid_relpath "$run" && [[ $run != */* ]] || die "invalid run name '$run' (a directory name inside the dataset)"
  done

  local run_id="${OPT_RUN_ID:-$(utc_stamp)}"
  check_run_id "$run_id"
  local started
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local common=(--role "$OPT_ROLE" --grid-cols "$OPT_COLS" --grid-rows "$OPT_ROWS")
  local fetch_flags=() run_flags=()
  if [[ $OPT_FORCE -eq 1 ]] || compgen -G "$ds_dir/cowtracker-logs/$run_id-summary*.json" >/dev/null; then
    fetch_flags+=(--refetch)
    run_flags+=(--force)
  fi
  if [[ $OPT_FORCE -eq 1 ]]; then fetch_flags+=(--force); fi
  if [[ $OPT_NO_PLACE -eq 1 ]]; then fetch_flags+=(--no-place); fi
  local extra=()
  if [[ ${#ADAPTER_EXTRA[@]} -gt 0 ]]; then extra=(-- "${ADAPTER_EXTRA[@]}"); fi

  build_adapter_argv "" "" "" ""
  local vflags=("${common[@]}")
  if [[ -z $EXPECT_POINTS ]]; then vflags+=(--any-points); fi

  log "== dataset $ds_key: ${#runs[@]} run(s), run id $run_id =="
  local entries vdir
  make_tmp entries; make_tmp vdir dir

  local failed=0 t0 status detail res
  for run in "${runs[@]}"; do
    t0=$SECONDS; status=ok; detail=""
    log ""
    log "---- $ds_key/$run ----"
    RUN_DIR="$ds_dir/$run"; RUN_KEY="$(basename -- "$ds_dir")/$run"
    res="$(local_results "$run_id")/tracks.json"
    if [[ ! -f $RUN_DIR/frames.json ]]; then
      status=failed; detail="no frames.json"
    elif [[ -s $res ]] && "$SELF" "${PASS_ARGS[@]}" validate "$RUN_DIR" --run-id "$run_id" "${vflags[@]}" >"$vdir/$run.json" 2>/dev/null; then
      status=skipped; detail="results for run id $run_id already present and valid"
      log "skip: $detail"
    else
      : >"$vdir/$run.json"
      if target_is_remote && [[ ! -e $(local_results "$run_id") ]] && t_has_results "$(result_dir "$run_id")"; then
        log "resuming: tracking already finished on $(t_where), fetching"
      elif ! "$SELF" "${PASS_ARGS[@]}" run "$RUN_DIR" "${common[@]}" --gpu "$OPT_GPU" --run-id "$run_id" "${run_flags[@]}" "${extra[@]}"; then
        status=failed; detail="run failed"
      fi
      if [[ $status == ok ]] && ! "$SELF" "${PASS_ARGS[@]}" fetch "$RUN_DIR" --run-id "$run_id" "${fetch_flags[@]}"; then
        status=failed; detail="fetch failed"
      fi
      if [[ $status == ok ]]; then
        if ! "$SELF" "${PASS_ARGS[@]}" validate "$RUN_DIR" --run-id "$run_id" "${vflags[@]}" >"$vdir/$run.json"; then
          status=failed; detail="validation failed"
          cat "$vdir/$run.json"
        fi
      fi
    fi
    if [[ $status == failed ]]; then
      failed=$((failed + 1))
      warn "$ds_key/$run: $detail"
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$run" "$status" "$detail" "$((SECONDS - t0))" "$vdir/$run.json" >>"$entries"
  done

  # Summary next to the data: <dataset>/cowtracker-logs/<run id>-summary.json
  local sdir="$ds_dir/cowtracker-logs" summary n=1
  mkdir -p -- "$sdir"
  summary="$sdir/$run_id-summary.json"
  while [[ -e $summary ]]; do n=$((n + 1)); summary="$sdir/$run_id-summary-$n.json"; done
  adapter_info
  "$TRACKER_LOCAL_PYTHON" "$LIB_DIR/run_info.py" summary --out "$summary" --run-id "$run_id" \
    --entries "$entries" --meta "dataset=$ds_key" "mode=$TRACKER_MODE" \
    "host=${TRACKER_SSH_HOST:-local}" "gpu_index=$OPT_GPU" "role=$OPT_ROLE" \
    "started=$started" "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "adapter_version=${ADAPTER_VERSION:-unknown}" "adapter_sha256=$ADAPTER_SHA256" \
    "cowtracker_commit=$COWTRACKER_COMMIT" "checkpoint_sha256=$CHECKPOINT_SHA256"
  log ""
  log "== summary: $summary =="
  cut -f1,2,3 "$entries" | sed 's/\t/  /g'
  if [[ $failed -gt 0 ]]; then
    die "$failed of ${#runs[@]} run(s) failed (re-run with --run-id $run_id to resume; finished runs are skipped)"
  fi
}
