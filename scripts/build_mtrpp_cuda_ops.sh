#!/usr/bin/env bash
# Build the MTR++ custom CUDA extensions (knn_cuda, attention_cuda) in place.
#
# These kernels back MTR++'s local-attention path. Without them the adapter
# falls back to global attention, which does not reproduce the paper.
# Must run on the machine that will train (Colab), since the extensions are
# ABI-bound to the installed torch + CUDA versions.
#
# Usage: bash scripts/build_mtrpp_cuda_ops.sh [upstream_dir]

set -euo pipefail

UPSTREAM="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/external/mtrpp}"
MAX_JOBS="${MAX_JOBS:-$(python - <<'PY'
import os
print(max(1, min(8, os.cpu_count() or 2)))
PY
)}"
export MAX_JOBS

echo "== environment =="
python -c "import torch; print('torch      :', torch.__version__); print('torch cuda :', torch.version.cuda); print('available  :', torch.cuda.is_available())"
nvcc --version | tail -2 || { echo "nvcc not found - install the CUDA toolkit"; exit 1; }

if ! python -c "import ninja" >/dev/null 2>&1; then
    echo "== installing ninja =="
    python -m pip install -q ninja
fi
python -c "import ninja; print('ninja     :', ninja.__file__)"

# nvcc and torch must agree on the CUDA major version, or the extension will
# build but fail to load with an undefined-symbol error.
python - <<'PY'
import re, subprocess, sys, torch
nvcc = subprocess.check_output(["nvcc", "--version"], text=True)
m = re.search(r"release (\d+)\.(\d+)", nvcc)
if not m:
    sys.exit("could not parse nvcc version")
nvcc_ver, torch_ver = m.group(1, 2), (torch.version.cuda or "").split(".")
if not torch_ver or torch_ver[0] != nvcc_ver[0]:
    sys.exit(f"CUDA major mismatch: nvcc {'.'.join(nvcc_ver)} vs torch {torch.version.cuda}")
print(f"cuda match : nvcc {'.'.join(nvcc_ver)} / torch {torch.version.cuda}")
PY

# Target the architectures of the GPUs actually present.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    TORCH_CUDA_ARCH_LIST="$(python -c "
import torch
archs = {'%d.%d' % torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
print(';'.join(sorted(archs)) or '7.0;7.5;8.0;8.6')
")"
    export TORCH_CUDA_ARCH_LIST
fi
echo "arch list  : $TORCH_CUDA_ARCH_LIST"
echo "max jobs   : $MAX_JOBS"

echo "== building in $UPSTREAM =="
cd "$UPSTREAM"
rm -rf build
python setup.py build_ext --inplace

echo "== verifying =="
cd "$UPSTREAM"
python -c "
import mtr.ops.knn.knn_cuda as k, mtr.ops.attention.attention_cuda as a
print('knn_cuda      :', k.__file__)
print('attention_cuda:', a.__file__)
print('knn_batch_mlogk present:', hasattr(k, 'knn_batch_mlogk'))
"
echo "OK - local attention path is now available."
