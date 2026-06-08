param(
    [int[]]$Ports = @(8090, 8091, 8502)
)

$ErrorActionPreference = "SilentlyContinue"

foreach ($port in $Ports) {
    $connections = Get-NetTCPConnection -LocalPort $port -State Listen
    if (-not $connections) {
        Write-Host "Port ${port}: free"
        continue
    }
    $pids = $connections.OwningProcess | Sort-Object -Unique
    foreach ($processId in $pids) {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $processId"
        Write-Host "Stopping PID ${processId} on port ${port}"
        if ($proc.CommandLine) {
            Write-Host "  $($proc.CommandLine)"
        }
        Stop-Process -Id $processId -Force
    }
}

Write-Host "Done. Restart with: .\scripts\start_translation_stack.ps1"