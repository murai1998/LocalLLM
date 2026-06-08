param(
    [string]$Quantization = "q6_k"
)

$ErrorActionPreference = "Stop"
$LocalRoot = Split-Path $PSScriptRoot -Parent

function Get-VenvPrefix([string]$Root) {
    $activate = Join-Path $Root ".venv\Scripts\Activate.ps1"
    if (Test-Path $activate) {
        return ". '$activate'; "
    }
    return ""
}

function Test-Health([string]$Url) {
    try {
        Invoke-RestMethod -Uri $Url -TimeoutSec 4 | Out-Null
        return $true
    } catch {
        return $false
    }
}

$env:LOCALLLM_MODEL__QUANTIZATION = $Quantization

Write-Host "LocalLLM root: $LocalRoot"
Write-Host "Quantization:  $Quantization"

if (Test-Health "http://127.0.0.1:8090/health") {
    Write-Host "[skip] LocalLLM gateway already healthy on :8090"
} else {
    $prefix = Get-VenvPrefix $LocalRoot
    Write-Host "[start] localllm-serve"
    Start-Process pwsh -ArgumentList @(
        "-NoExit",
        "-Command",
        "${prefix}Set-Location '$LocalRoot'; localllm-serve"
    )
}

Write-Host ""
Write-Host "UI: localllm-streamlit  -> Chat | Agent | Translate"
Write-Host "    localllm-translate-streamlit  -> opens Translate mode directly"