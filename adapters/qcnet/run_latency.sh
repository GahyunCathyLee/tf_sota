#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/gahyun/miniconda3/envs/tf/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

WARMUP="${WARMUP:-1000}"
ITERS="${ITERS:-10000}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/latency/qcnet}"
mkdir -p "$LOG_DIR"

# Optional data override: DATA_ROOT=/path/to/sota_experiments_data ./adapters/qcnet/run_latency.sh
# Optional checkpoint overrides:
#   CKPT_ROOT=/path/to/qcnet_ckpts ./adapters/qcnet/run_latency.sh
#   EXID_BASE_CKPT=/path/to/exiD0-5.pt EXID_I_CKPT=/path/to/exiD2-5.pt ./adapters/qcnet/run_latency.sh

cases=(
  "exiD-baseline|ckpts/qcnet/exiD0-5/best.pt"
  "exiD-+I|ckpts/qcnet/exiD2-5/best.pt"
  "highD-baseline|ckpts/qcnet/highD0-4/best.pt"
  "highD-+I|ckpts/qcnet/highD2-3/best.pt"
)

cd "$ROOT"
for row in "${cases[@]}"; do
  IFS='|' read -r name ckpt <<< "$row"
  dataset="${name%%-*}"
  condition="${name#*-}"
  log_path="${LOG_DIR}/${name}.log"

  ckpt_key=""
  if [[ "$dataset" == "exiD" && "$condition" == "baseline" ]]; then
    ckpt_key="${EXID_BASE_CKPT:-}"
  elif [[ "$dataset" == "exiD" ]]; then
    ckpt_key="${EXID_I_CKPT:-}"
  elif [[ "$dataset" == "highD" && "$condition" == "baseline" ]]; then
    ckpt_key="${HIGHD_BASE_CKPT:-}"
  elif [[ "$dataset" == "highD" ]]; then
    ckpt_key="${HIGHD_I_CKPT:-}"
  fi
  if [[ -n "$ckpt_key" ]]; then
    ckpt="$ckpt_key"
  elif [[ -n "${CKPT_ROOT:-}" ]]; then
    ckpt="${CKPT_ROOT}/${ckpt#ckpts/qcnet/}"
  fi

  if [[ ! -f "$ckpt" ]]; then
    echo "[SKIP] ${name}: missing ${ckpt}"
    continue
  fi

  cmd=(
    "$PYTHON_BIN" evaluate.py
    --model qcnet
    --ckpt "$ckpt"
    --split test
    --measure-time
    --warmup "$WARMUP"
    --iters "$ITERS"
  )
  if [[ -n "${DATA_ROOT:-}" ]]; then
    cmd+=(--data-root "$DATA_ROOT")
  fi

  echo "[RUN] qcnet ${name}"
  printf '  %q' "${cmd[@]}"
  echo
  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi
  "${cmd[@]}" 2>&1 | tee "$log_path"
done
