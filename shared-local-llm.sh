#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SHARED_LOCAL_LLM_ENV:-$SCRIPT_DIR/shared-local-llm.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

ACTION="${1:-status}"
TARGET="${2:-finance}"
PROFILE="${LOCAL_LLM_RUNTIME_PROFILE:-mlx-vlm-finance-bf16}"

case "$TARGET" in
  finance|ocr|vision|text|qwen) ;;
  *)
    echo "Unknown runtime target: $TARGET" >&2
    exit 2
    ;;
esac

case "$ACTION" in
  start)
    exec "$SCRIPT_DIR/.venv/bin/llm-control" start "$PROFILE"
    ;;
  stop)
    exec "$SCRIPT_DIR/.venv/bin/llm-control" stop
    ;;
  status)
    exec "$SCRIPT_DIR/.venv/bin/llm-control" status
    ;;
  profiles)
    exec "$SCRIPT_DIR/.venv/bin/llm-control" profiles
    ;;
  *)
    echo "Usage: $0 {start|stop|status|profiles} [finance|ocr|vision|text|qwen]" >&2
    exit 2
    ;;
esac
