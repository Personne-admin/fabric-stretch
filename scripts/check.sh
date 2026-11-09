#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -x .venv/bin/python ]]; then
    PATH="$PWD/.venv/bin:$PATH"
fi

run_python() {
    echo "== python"
    ruff check .
    ruff format --check .
    python -m pytest -q
}

run_rust() {
    echo "== rust"
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo clippy --workspace --all-targets --features fabric-viewer/demo -- -D warnings
    cargo test --workspace
}

run_shell() {
    echo "== shellcheck"
    if ! command -v shellcheck >/dev/null 2>&1; then
        echo "shellcheck not installed; skipping" >&2
        return 0
    fi
    shellcheck -x server/bootstrap.sh server/fabsim-track server/lib/*.sh server/tests/*.sh \
        server/tests/fakebin/* scripts/*.sh
}

targets=("$@")
if [[ ${#targets[@]} -eq 0 ]]; then
    targets=(python rust shell)
fi

for target in "${targets[@]}"; do
    case "$target" in
        python) run_python ;;
        rust) run_rust ;;
        shell) run_shell ;;
        *) echo "unknown target: $target (use python, rust or shell)" >&2; exit 2 ;;
    esac
done
echo "all checks passed"
