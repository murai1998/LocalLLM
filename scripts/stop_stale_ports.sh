#!/usr/bin/env bash
# Free the ports used by the LocalLLM stack by stopping stale LISTEN-ers.
# macOS/Linux counterpart of stop_stale_ports.ps1.
#
#   8080  llama-server (inference)
#   8090  localllm-serve (gateway)
#   8091  reserved (whisper sidecar)
#   8095  localllm-webui (web UI)
#   8501  streamlit chat
#   8502  streamlit translate
#
# Lingering TIME_WAIT / FIN_WAIT sockets have no owning process and clear on
# their own within ~30s — only LISTEN holders block a restart, so only those
# are targeted.
#
# Usage:
#   ./scripts/stop_stale_ports.sh                # all stack ports
#   ./scripts/stop_stale_ports.sh 8095           # specific port(s)
#   ./scripts/stop_stale_ports.sh --dry-run      # show what would be killed

set -u

DRY_RUN=0
PORTS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run|-n) DRY_RUN=1 ;;
        *[!0-9]*) echo "Unknown argument: $arg" >&2; exit 2 ;;
        *) PORTS+=("$arg") ;;
    esac
done
if [ ${#PORTS[@]} -eq 0 ]; then
    PORTS=(8080 8090 8091 8095 8501 8502)
fi

failures=0

for port in "${PORTS[@]}"; do
    pids=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | sort -u)
    if [ -z "$pids" ]; then
        echo "Port ${port}: free"
        continue
    fi

    for pid in $pids; do
        if [ "$pid" -eq $$ ]; then
            echo "Port ${port}: held by this shell — skipping"
            continue
        fi
        cmd=$(ps -p "$pid" -o command= 2>/dev/null || echo "unknown")
        echo "Port ${port}: stopping PID ${pid}"
        echo "  ${cmd}"
        if [ "$DRY_RUN" -eq 1 ]; then
            echo "  (dry run — not killed)"
            continue
        fi
        if ! kill "$pid" 2>/dev/null; then
            echo "  FAILED to stop PID ${pid}" >&2
            failures=$((failures + 1))
        fi
    done

    if [ "$DRY_RUN" -eq 0 ]; then
        # Confirm the listener is actually gone (kill is async); escalate to
        # SIGKILL if it ignores SIGTERM.
        deadline=$((SECONDS + 5))
        while [ "$SECONDS" -lt "$deadline" ]; do
            still=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null)
            [ -z "$still" ] && break
            sleep 0.25
        done
        if [ -n "${still:-}" ]; then
            for pid in $still; do
                kill -9 "$pid" 2>/dev/null
            done
            sleep 0.5
            still=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null)
        fi
        if [ -n "${still:-}" ]; then
            echo "Port ${port}: STILL HELD after kill — check permissions" >&2
            failures=$((failures + 1))
        else
            echo "Port ${port}: freed"
        fi
    fi
done

if [ "$failures" -gt 0 ]; then
    echo ""
    echo "Done with ${failures} failure(s)." >&2
    exit 1
fi
echo ""
echo "Done. Restart with: localllm-serve / localllm-webui (or ./scripts/start_translation_stack.sh)"
