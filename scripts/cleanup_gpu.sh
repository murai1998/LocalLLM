#!/usr/bin/env bash
# Stop LocalLLM / llama-server processes and free GPU/Metal memory.
#
# Usage:
#   ./scripts/cleanup_gpu.sh
#   ./scripts/cleanup_gpu.sh --streamlit
#   ./scripts/cleanup_gpu.sh --dry-run

set -euo pipefail

INCLUDE_STREAMLIT=false
DRY_RUN=false

for arg in "$@"; do
  case "$arg" in
    --streamlit) INCLUDE_STREAMLIT=true ;;
    --dry-run) DRY_RUN=true ;;
    -h|--help)
      echo "Usage: $0 [--streamlit] [--dry-run]"
      exit 0
      ;;
  esac
done

PORTS=(8080 8090)
if $INCLUDE_STREAMLIT; then
  PORTS+=(8501 8502)
fi

PATTERNS=(
  "llama-server"
  "localllm-serve"
  "localllm.service"
  "whisper-stt-serve"
  "streamlit"
)

step() { echo ""; echo "==> $1"; }

gpu_snapshot() {
  step "$1"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    echo ""
    echo "Compute processes:"
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || echo "  (none)"
  else
    echo "nvidia-smi not found."
    if [[ "$(uname -s)" == "Darwin" ]]; then
      echo "Metal: stop llama-server manually if VRAM/unified memory is stuck."
    fi
  fi
}

kill_pid() {
  local pid="$1"
  local reason="$2"
  [[ "$pid" -le 4 ]] && return 0
  echo "Stop PID $pid — $reason"
  if $DRY_RUN; then
    echo "  [dry-run] would kill"
  else
    kill -9 "$pid" 2>/dev/null || true
  fi
}

kill_by_pattern() {
  step "Stopping by process pattern"
  for pattern in "${PATTERNS[@]}"; do
    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    for pid in $pids; do
      kill_pid "$pid" "matches $pattern"
    done
  done
}

kill_by_port() {
  step "Stopping listeners on stack ports"
  for port in "${PORTS[@]}"; do
    if command -v lsof >/dev/null 2>&1; then
      pids=$(lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null || true)
      if [[ -z "$pids" ]]; then
        echo "Port $port: free"
        continue
      fi
      for pid in $pids; do
        kill_pid "$pid" "listening on port $port"
      done
    else
      echo "lsof not found — skip port $port"
    fi
  done
}

gpu_snapshot "GPU before cleanup"
kill_by_pattern
kill_by_port
sleep 1
gpu_snapshot "GPU after cleanup"

echo ""
if $DRY_RUN; then
  echo "Dry-run mode — no processes were killed."
else
  echo "Done. Restart with: localllm-serve"
fi