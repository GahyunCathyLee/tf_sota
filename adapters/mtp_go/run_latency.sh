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
LOG_DIR="${LOG_DIR:-${ROOT}/logs/latency/mtp_go}"
mkdir -p "$LOG_DIR"

cases=(
  "exiD-baseline|ckpts/mtp_go/exiD0-5/best.ckpt"
  "exiD-+I|ckpts/mtp_go/exiD2-5/best.ckpt"
  "highD-baseline|ckpts/mtp_go/highD0-4/best.ckpt"
  "highD-+I|ckpts/mtp_go/highD2-3/best.ckpt"
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
    --model mtp_go
    --ckpt "$ckpt"
    --split test
    --measure-time
    --warmup "$WARMUP"
    --iters "$ITERS"
  )

  echo "[RUN] mtp_go ${name}"
  printf '  %q' "${cmd[@]}"
  echo
  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi
  "${cmd[@]}" 2>&1 | tee "$log_path"
done
