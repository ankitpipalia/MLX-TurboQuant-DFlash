#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
TARGET_DIR="${QWEN38_DFLASH2_MODEL:-$HOME/Models/mlx/mlx-community--Qwen3.8-27B-4bit}"
DRAFT_DIR="${QWEN38_DFLASH2_DRAFT:-$HOME/Models/mlx/ProCreations--Qwen3.8-27B-DFlash2-MLXFast-Q4}"
TARGET_REV="3e6447f082e89cc7f0bc6e5441afd38dfce760ff"
DRAFT_REV="3485116dcbd1e94715036193b57aec4841c60d59"

# Both artifacts are MLX affine 4-bit. The draft is proposal-only and must be
# paired with the exact Qwen3.8-27B target family for target-side verification.
HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY=false \
HF_XET_FIXED_DOWNLOAD_CONCURRENCY=32 \
HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY=32 \
HF_XET_NUM_CONCURRENT_RANGE_GETS=32 \
HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS=3 \
hf download mlx-community/Qwen3.8-27B-4bit \
  --revision "$TARGET_REV" --local-dir "$TARGET_DIR"

hf download ProCreations/Qwen3.8-27B-DFlash2-MLXFast-Q4 \
  --revision "$DRAFT_REV" --local-dir "$DRAFT_DIR"

# The compact draft repository intentionally publishes only its strict-loaded
# MLX tensor tree. Install the matching upstream architecture metadata so the
# standard dflash-mlx loader can reconstruct and quantize the correct modules.
install -m 0644 "$SCRIPT_DIR/../config/qwen38-dflash2-q4-config.json" \
  "$DRAFT_DIR/config.json"

(
  cd "$TARGET_DIR"
  shasum -a 256 -c "$SCRIPT_DIR/../config/qwen38-dflash2-target-sha256.txt"
)
printf '%s  %s\n' \
  "95cd528f14ced30c4cb4933c2ae233f822c20cc6c38f7502adccef359c0141ce" \
  "$DRAFT_DIR/model.safetensors" | shasum -a 256 -c -

printf '%s\n' "Qwen3.8 target: $TARGET_DIR" "DFlash2 draft: $DRAFT_DIR"
