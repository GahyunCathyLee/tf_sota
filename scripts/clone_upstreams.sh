#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${EXPERIMENT_DIR}/external"

mkdir -p "${EXTERNAL_DIR}"

clone_if_missing() {
  local name="$1"
  local url="$2"
  local dst="${EXTERNAL_DIR}/${name}"

  if [[ -d "${dst}/.git" ]]; then
    echo "[skip] ${name}: already cloned at ${dst}"
    return
  fi

  echo "[clone] ${name}: ${url}"
  git clone "${url}" "${dst}"
}

clone_if_missing "hivt" "https://github.com/ZikangZhou/HiVT.git"
clone_if_missing "qcnet" "https://github.com/ZikangZhou/QCNet.git"
clone_if_missing "mtrpp" "https://github.com/sshaoshuai/MTR.git"
clone_if_missing "par" "https://github.com/neerjathakkar/PAR.git"
clone_if_missing "simpl" "https://github.com/HKUST-Aerial-Robotics/SIMPL.git"
clone_if_missing "hptr" "https://github.com/zhejz/HPTR.git"
clone_if_missing "gameformer" "https://github.com/MCZhi/GameFormer.git"
clone_if_missing "mtp_go" "https://github.com/westny/mtp-go.git"
clone_if_missing "trajectronpp" "https://github.com/StanfordASL/Trajectron-plus-plus.git"

cat <<'MSG'

MotionLM is not cloned because no official public training repository was found.
QCNeXt is tracked through the QCNet repository and will need a dedicated adapter.
MSG
