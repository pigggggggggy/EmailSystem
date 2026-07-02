#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANGELSLIM_DIR="${ANGELSLIM_DIR:-${ROOT_DIR}/third_party/AngelSlim}"
ANGELSLIM_REPO="${ANGELSLIM_REPO:-https://github.com/Tencent/AngelSlim.git}"
ANGELSLIM_COMMIT="${ANGELSLIM_COMMIT:-41f487f8c68985a7853b249f15729e2f7193dbb5}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ ! -d "$ANGELSLIM_DIR/.git" ]]; then
  mkdir -p "$(dirname "$ANGELSLIM_DIR")"
  git clone "$ANGELSLIM_REPO" "$ANGELSLIM_DIR"
fi
git -C "$ANGELSLIM_DIR" fetch origin "$ANGELSLIM_COMMIT"
git -C "$ANGELSLIM_DIR" checkout --detach "$ANGELSLIM_COMMIT"
"$PYTHON_BIN" "$ROOT_DIR/training/patch_angelslim_attention.py" --angelslim-dir "$ANGELSLIM_DIR"
"$PYTHON_BIN" -m pip install -e "${ANGELSLIM_DIR}[speculative]"
echo "AngelSlim ready: $ANGELSLIM_DIR @ $ANGELSLIM_COMMIT"
