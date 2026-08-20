#!/bin/sh
set -eu

TARGET_DIR="${QWEN38_MTP_MODEL:-$HOME/Models/mlx/EigenLabs--Qwen3.8-27B-4bit}"
HEAD_DIR="${QWEN38_MTP_HEAD:-$HOME/Models/mlx/lowskillcoding--qwen38-mtp-head-4bit-g64}"
TARGET_REV="eda45ab47f465d08d6558f0353a2346e2eb9d5b3"
HEAD_REV="0966ddaff972fd3ca2be08f3640603b47e9ce70a"

HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY=false \
HF_XET_FIXED_DOWNLOAD_CONCURRENCY=32 \
HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY=32 \
HF_XET_NUM_CONCURRENT_RANGE_GETS=32 \
HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS=3 \
hf download EigenLabs/Qwen3.8-27B-4bit \
  --revision "$TARGET_REV" --local-dir "$TARGET_DIR"
hf download lowskillcoding/qwen38-mtp-head-4bit-g64 model.safetensors \
  --revision "$HEAD_REV" --local-dir "$HEAD_DIR"

"${MLX_PYTHON_BIN:-$(dirname "$0")/../.venv/bin/python}" \
  "$(dirname "$0")/prepare_qwen38_mtp_head.py" \
  --target "$TARGET_DIR" --head "$HEAD_DIR"
