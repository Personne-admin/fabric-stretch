# shellcheck shell=bash

init_target() {
  SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15)
  if [[ -n $TRACKER_SSH_KEY ]]; then
    SSH_OPTS+=(-i "$TRACKER_SSH_KEY" -o IdentitiesOnly=yes)
  fi

  if [[ -n $TRACKER_SSH_JUMP ]]; then
    SSH_OPTS+=(-J "$TRACKER_SSH_JUMP")
  fi

  if [[ -n $TRACKER_SSH_OPTS ]]; then
    local extra
    read -r -a extra <<<"$TRACKER_SSH_OPTS"
    SSH_OPTS+=("${extra[@]}")
  fi

  RSYNC_SSH="$(quote_words ssh "${SSH_OPTS[@]}")"
}

target_is_remote() { [[ $TRACKER_MODE == ssh ]]; }

# t_sh [ARG...]: run the script on stdin on the target, with ARG as $1...
t_sh() {
  if target_is_remote; then
    # ssh joins its arguments and the remote shell parses them again: quote.
    ssh "${SSH_OPTS[@]}" -- "$TRACKER_SSH_HOST" "bash -s -- $(quote_words "$@")"
  else
    bash -s -- "$@"
  fi
}

# t_where: human-readable target name for messages.
t_where() {
  if target_is_remote; then printf '%s' "$TRACKER_SSH_HOST"; else printf 'this machine'; fi
}

# t_remote_path PATH: PATH as seen by rsync (host:path for ssh).
t_remote_path() {
  if target_is_remote; then printf '%s:%s' "$TRACKER_SSH_HOST" "$1"; else printf '%s' "$1"; fi
}

# t_exists PATH: succeeds when PATH exists on the target.
t_exists() {
  # shellcheck disable=SC2016  # $1 is for the script on the target
  t_sh "$1" <<<'test -e "$1"'
}

# t_has_results DIR: succeeds when DIR/tracks.json exists on the target.
t_has_results() {
  # shellcheck disable=SC2016
  t_sh "$1" <<<'test -s "$1/tracks.json"'
}

# t_mkdir PATH...
t_mkdir() {
  t_sh "$@" <<<'mkdir -p -- "$@"'
}

# t_put_file LOCAL TARGET_PATH: copy one file (creating its directory).
t_put_file() {
  local src="$1" dest="$2"
  t_mkdir "$(dirname "$dest")" || return 1
  if target_is_remote; then
    rsync -t -e "$RSYNC_SSH" -- "$src" "$(t_remote_path "$dest")"
  else
    install -m 0644 -- "$src" "$dest"
  fi
}

# t_write_text TARGET_PATH: write stdin to a file on the target.
t_write_text() {
  if target_is_remote; then
    ssh "${SSH_OPTS[@]}" -- "$TRACKER_SSH_HOST" "cat > $(printf '%q' "$1")"
  else
    cat > "$1"
  fi
}

# t_pull_dir TARGET_DIR LOCAL_DIR: copy a directory tree from the target.
t_pull_dir() {
  mkdir -p -- "$2"
  rsync -a --partial -e "$RSYNC_SSH" -- "$(t_remote_path "$1")/" "$2/"
}

# t_push_files LOCAL_BASE FILELIST TARGET_DIR: copy exactly the files named in
# FILELIST (paths relative to LOCAL_BASE, one per line) into TARGET_DIR.
t_push_files() {
  local base="$1" list="$2" dest="$3"
  t_mkdir "$dest" || return 1
  rsync -a --partial -e "$RSYNC_SSH" --files-from="$list" -- "$base/" "$(t_remote_path "$dest")/"
}

# adapter_info: set ADAPTER_SHA256, ADAPTER_VERSION and ADAPTER_COMMIT.
adapter_info() {
  [[ -f $FABSIM_ADAPTER ]] || die "adapter not found: $FABSIM_ADAPTER (set FABSIM_ADAPTER)"
  ADAPTER_SHA256="$(sha256sum "$FABSIM_ADAPTER" | cut -d' ' -f1)"
  ADAPTER_VERSION="$(sed -n 's/^ADAPTER_VERSION = "\(.*\)"$/\1/p' "$FABSIM_ADAPTER" | head -n1)"
  ADAPTER_COMMIT=""
  local dir
  dir="$(dirname "$FABSIM_ADAPTER")"
  if ADAPTER_COMMIT="$(git -C "$dir" rev-parse --short=12 HEAD 2>/dev/null)"; then
    if [[ -n $(git -C "$dir" status --porcelain -- "$FABSIM_ADAPTER" 2>/dev/null) ]]; then
      ADAPTER_COMMIT+="-dirty"
    fi
  else
    ADAPTER_COMMIT=""
  fi
}

# upload_adapter: copy the adapter to $TRACKER_ROOT/code/adapter/ and write
# cowtracker_to_fabsim.version (sha256, version, commit) beside it.
upload_adapter() {
  adapter_info
  local dest stamp
  dest="$(root_adapter)"
  stamp="${dest%.py}.version"
  log "== adapter =="
  log "  file    : $FABSIM_ADAPTER"
  log "  version : ${ADAPTER_VERSION:-unknown} (sha256 ${ADAPTER_SHA256:0:12}, commit ${ADAPTER_COMMIT:-unknown})"
  t_put_file "$FABSIM_ADAPTER" "$dest" || die "could not upload the adapter to $(t_where)"
  printf 'version=%s\nsha256=%s\ncommit=%s\nuploaded=%s\n' \
    "${ADAPTER_VERSION:-unknown}" "$ADAPTER_SHA256" "${ADAPTER_COMMIT:-}" "$(utc_stamp)" |
    t_write_text "$stamp" || die "could not write the adapter version stamp"
}
