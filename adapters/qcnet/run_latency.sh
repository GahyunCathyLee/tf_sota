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

cases=(
  "exiD-baseline|ckpts/qcnet/exiD0-5/best.pt"
  "exiD-+I|ckpts/qcnet/exiD2-5/best.pt"
  "highD-baseline|ckpts/qcnet/highD0-4/best.pt"
  "highD-+I|ckpts/qcnet/highD2-3/best.pt"
)

cd "$ROOT"
for row in "${cases[@]}"; do
  IFS='|' read -r name ckpt <<< "$row"
  log_path="${LOG_DIR}/${name}.log"

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

  echo "[RUN] qcnet ${name}"
  printf '  %q' "${cmd[@]}"
  echo
  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi
  "${cmd[@]}" 2>&1 | tee "$log_path"
done
