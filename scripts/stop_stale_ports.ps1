<#
.SYNOPSIS
  Free the ports used by the LocalLLM stack by stopping stale listeners.

.DESCRIPTION
  Kills processes LISTENING on the stack's ports:
    8080  llama-server (inference)
    8090  localllm-serve (gateway)
    8091  reserved (whisper sidecar)
    8095  localllm-webui (web UI)
    8501  streamlit chat
    8502  streamlit translate
  Lingering sockets in TIME_WAIT / FIN_WAIT have no owning process and clear on
  their own within ~30s — only LISTEN holders block a restart, so only those
  are targeted.

.EXAMPLE
  .\scripts\stop_stale_ports.ps1            # all stack ports
  .\scripts\stop_stale_ports.ps1 -Ports 8095
  .\scripts\stop_stale_ports.ps1 -WhatIfOnly  # show what would be killed
#>
param(
    [int[]]$Ports = @(8080, 8090, 8091, 8095, 8501, 8502),
    [switch]$WhatIfOnly
)

$failures = 0

foreach ($port in $Ports) {
    $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if (-not $connections) {
        Write-Host "Port ${port}: free"
        continue
    }

    $listenerPids = $connections.OwningProcess | Sort-Object -Unique
    foreach ($processId in $listenerPids) {
        # PID 0 (Idle) and 4 (System) own kernel sockets; never touch them.
        if ($processId -le 4) {
            Write-Host "Port ${port}: held by system PID ${processId} — skipping" -ForegroundColor Yellow
            continue
        }
        if ($processId -eq $PID) {
            Write-Host "Port ${port}: held by this PowerShell process — skipping" -ForegroundColor Yellow
            continue
        }

        $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction SilentlyContinue
        $name = if ($proc) { $proc.Name } else { "unknown" }
        Write-Host "Port ${port}: stopping PID ${processId} (${name})"
        if ($proc -and $proc.CommandLine) {
            Write-Host "  $($proc.CommandLine)"
        }

        if ($WhatIfOnly) {
            Write-Host "  (dry run — not killed)" -ForegroundColor Yellow
            continue
        }

        try {
            Stop-Process -Id $processId -Force -ErrorAction Stop
        } catch {
            Write-Host "  FAILED to stop PID ${processId}: $($_.Exception.Message)" -ForegroundColor Red
            $failures++
            continue
        }
    }

    if (-not $WhatIfOnly) {
        # Confirm the listener is actually gone (kill is async).
        $deadline = (Get-Date).AddSeconds(5)
        do {
            Start-Sleep -Milliseconds 250
            $still = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        } while ($still -and (Get-Date) -lt $deadline)

        if ($still) {
            Write-Host "Port ${port}: STILL HELD after kill — run as Administrator?" -ForegroundColor Red
            $failures++
        } else {
            Write-Host "Port ${port}: freed" -ForegroundColor Green
        }
    }
}

if ($failures -gt 0) {
    Write-Host "`nDone with $failures failure(s)." -ForegroundColor Red
    exit 1
}
Write-Host "`nDone. Restart with: localllm-serve / localllm-webui (or .\scripts\start_translation_stack.ps1)"
