#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
dry_run=false
cli_args=()

usage() {
    cat <<'HELP'
Infer a mask from the current frame and an externally warped past frame.

Usage:
  bash scripts/infer.sh --checkpoint /runs/model/last.pth \
    --current /data/current.png --past /data/past_warped.png --output /results/mask.png

Optional environment configuration:
  PYTHON_BIN       Python executable in a PyTorch environment (default: python3)
  CHECKPOINT       Full ModelBuilder checkpoint
  CURRENT_FRAME    Current RGB/grayscale image
  PAST_FRAME       Past image already aligned into current-frame coordinates
  OUTPUT_MASK      Output mask PNG
  SEG_INPUT        Optional flow/flow_rgb checkpoint consistency check
  INFER_ITERS_S8   Optional iteration override; unset = checkpoint value

The head mode, resolution and iterations otherwise come from the checkpoint.
Built-in warp is disabled by default (--stabilization none).
--dry-run prints the command without inference.
--python-help shows all options accepted by tool/infer.py.
All other arguments are forwarded unchanged and override environment defaults.
HELP
}

for arg in "$@"; do
    case "$arg" in
        --help|-h) usage; exit 0 ;;
        --dry-run) dry_run=true ;;
        --python-help) cli_args+=(--help) ;;
        *) cli_args+=("$arg") ;;
    esac
done

command=("$python_bin" "$project_root/tool/infer.py" --stabilization none)
for mapping in CHECKPOINT:checkpoint CURRENT_FRAME:current PAST_FRAME:past OUTPUT_MASK:output \
               SEG_INPUT:seg_input INFER_ITERS_S8:iters_s8; do
    env_name="${mapping%%:*}"
    option="${mapping#*:}"
    value="${!env_name-}"
    if [[ -n "$value" ]]; then
        command+=("--$option" "$value")
    fi
done
command+=("${cli_args[@]}")

if "$dry_run"; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

if ! command -v "$python_bin" >/dev/null 2>&1; then
    printf 'Python executable not found: %s. Activate the PyTorch environment or set PYTHON_BIN.\n' "$python_bin" >&2
    exit 127
fi
exec "${command[@]}"
