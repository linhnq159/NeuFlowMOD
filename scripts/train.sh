#!/usr/bin/env bash
set -euo pipefail

# Use the active PyTorch environment, or set PYTHON_BIN=/path/to/python.
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
dry_run=false
cli_args=()

usage() {
    cat <<'HELP'
Train NeuFlow motion segmentation from frame/mask folders.

Usage:
  bash scripts/train.sh --data_path /data/train --seg_input flow
  bash scripts/train.sh --data_path /data/train --seg_input flow_rgb
  bash scripts/train.sh --data_path /data/train --seg_input flow_rgb --transfer /runs/flow/last.pth
  bash scripts/train.sh --data_path /data/train --resume /runs/rgb/last.pth

Optional environment configuration:
  PYTHON_BIN   Python executable in a PyTorch environment (default: python3)
  DATA_PATH    Dataset root containing images/ and masks/
  SEG_INPUT    flow or flow_rgb; unset = config default, or checkpoint when resuming
  WORK_DIR     Output root (default: <repository>/work_seg/checkpoint)
  RESUME       Checkpoint to resume, including optimizer and epoch
  TRANSFER     Checkpoint for weights-only transfer with a new optimizer

--dry-run prints the command without training.
--python-help shows all options accepted by tool/train.py.
All other arguments are forwarded unchanged and override environment defaults.
The Python trainer controls batch size, resolution, iterations, loss and device.
New runs default to training-time homography stabilization.
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

command=("$python_bin" "$project_root/tool/train.py"
         --work_dir "${WORK_DIR:-$project_root/work_seg/checkpoint}")
for mapping in DATA_PATH:data_path SEG_INPUT:seg_input RESUME:resume TRANSFER:transfer; do
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
