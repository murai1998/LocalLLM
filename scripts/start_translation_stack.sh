#!/usr/bin/env bash
set -euo pipefail

QUANT="${1:-q6_k}"
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export LOCALLLM_MODEL__QUANTIZATION="$QUANT"

health() {
  curl -sf "http://127.0.0.1:8090/health" >/dev/null 2>&1
}

echo "LocalLLM root: $LOCAL_ROOT"
echo "Quantization:  $QUANT"

if health; then
  echo "[skip] LocalLLM gateway already healthy on :8090"
else
  echo "[start] localllm-serve"
  (
    cd "$LOCAL_ROOT"
    exec localllm-serve
  ) &
fi

echo ""
echo "UI: localllm-streamlit  (Chat | Agent | Translate)"