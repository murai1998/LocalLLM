#!/usr/bin/env pwsh
<#
.SYNOPSIS
  Stop LocalLLM / llama-server GPU processes and free VRAM.

.DESCRIPTION
  - Shows GPU memory before and after
  - Stops llama-server and localllm gateway (ports 8080, 8090)
  - Stops orphaned Python processes that still hold GPU memory (nvidia-smi)
  - Optional Streamlit ports (8501, 8502)

.EXAMPLE
  .\scripts\cleanup_gpu.ps1
  .\scripts\cleanup_gpu.ps1 -IncludeStreamlit
  .\scripts\cleanup_gpu.ps1 -WhatIf
#>
param(
    [switch]$IncludeStreamlit,
    [switch]$WhatIf
)

$ErrorActionPreference = "SilentlyContinue"

$Ports = @(8080, 8090)
if ($IncludeStreamlit) {
    $Ports += @(8501, 8502)
}

$GpuProcessPatterns = @(
    "llama-server",
    "llama-server.exe",
    "localllm-serve",
    "localllm.service",
    "whisper-stt-serve"
)

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message"
}

function Show-GpuSnapshot([string]$Label) {
    Write-Step $Label
    $nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidia) {
        Write-Host "nvidia-smi not found (skip GPU stats)."
        return
    }
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    Write-Host ""
    Write-Host "Compute processes:"
    $apps = nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>$null
    if ($apps) {
        $apps | ForEach-Object { Write-Host "  $_" }
    } else {
        Write-Host "  (none)"
    }
}

function Get-ProcessCommandLine([int]$ProcessId) {
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    if ($proc) {
        return $proc.CommandLine
    }
    return ""
}

function Stop-ProcessSafe([int]$ProcessId, [string]$Reason) {
    if ($ProcessId -le 4) {
        return
    }
    $cmd = Get-ProcessCommandLine $ProcessId
    Write-Host "Stop PID $ProcessId — $Reason"
    if ($cmd) {
        Write-Host "  $cmd"
    }
    if ($WhatIf) {
        Write-Host "  [WhatIf] would stop"
        return
    }
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
}

function Stop-ByPort([int]$Port) {
    $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if (-not $connections) {
        Write-Host "Port ${Port}: free"
        return
    }
    foreach ($processId in ($connections.OwningProcess | Sort-Object -Unique)) {
        Stop-ProcessSafe -ProcessId $processId -Reason "listening on port $Port"
    }
}

function Stop-ByNamePatterns {
    foreach ($pattern in $GpuProcessPatterns) {
        Get-Process -Name $pattern -ErrorAction SilentlyContinue | ForEach-Object {
            Stop-ProcessSafe -ProcessId $_.Id -Reason "process name $pattern"
        }
    }

    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine } |
        ForEach-Object {
            foreach ($pattern in $GpuProcessPatterns) {
                if ($_.CommandLine -like "*$pattern*") {
                    Stop-ProcessSafe -ProcessId $_.ProcessId -Reason "command line matches $pattern"
                    break
                }
            }
        }
}

function Stop-GpuComputeOrphans {
    $nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidia) {
        return
    }

    $lines = nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>$null
    if (-not $lines) {
        Write-Host "No GPU compute processes reported."
        return
    }

    foreach ($line in $lines) {
        if ($line -notmatch "^\s*(\d+)\s*,\s*(.+)$") {
            continue
        }
        $processId = [int]$Matches[1]
        $name = $Matches[2].Trim()
        $cmd = Get-ProcessCommandLine $processId

        $isStackProcess = $false
        foreach ($pattern in $GpuProcessPatterns) {
            if ($name -like "*$pattern*" -or $cmd -like "*$pattern*") {
                $isStackProcess = $true
                break
            }
        }
        if ($name -match "python" -and $cmd -match "localllm|llama-server|whisperstt|streamlit") {
            $isStackProcess = $true
        }

        if ($isStackProcess) {
            Stop-ProcessSafe -ProcessId $processId -Reason "GPU compute orphan ($name)"
        }
    }
}

Show-GpuSnapshot "GPU before cleanup"

Write-Step "Stopping by process name / command line"
Stop-ByNamePatterns

Write-Step "Stopping listeners on stack ports"
foreach ($port in $Ports) {
    Stop-ByPort -Port $port
}

Start-Sleep -Seconds 1

Write-Step "Checking GPU compute orphans"
Stop-GpuComputeOrphans

Start-Sleep -Seconds 1
Show-GpuSnapshot "GPU after cleanup"

Write-Host ""
if ($WhatIf) {
    Write-Host "WhatIf mode — no processes were killed."
} else {
    Write-Host "Done. Restart with: localllm-serve"
}