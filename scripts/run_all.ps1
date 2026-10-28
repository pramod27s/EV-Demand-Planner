# Full ML pipeline (Windows PowerShell). Usage: .\scripts\run_all.ps1 [-Quick]
param([switch]$Quick)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)   # run from the project root
$py = if (Test-Path .venv\Scripts\python.exe) { ".venv\Scripts\python.exe" } else { "python" }

function Step([string[]]$cmd) {
    Write-Host ">> $py $($cmd -join ' ')" -ForegroundColor Green
    & $py @cmd
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $($cmd -join ' ')" }
}

Step @("-m", "src.data.preprocess")
Step @("-m", "src.experiments.run_benchmarks")
Step @("-m", "src.experiments.run_baselines")
if ($Quick) { Step @("-m", "src.experiments.run_tuning", "--quick") } else { Step @("-m", "src.experiments.run_tuning") }
Step @("-m", "src.experiments.summarize")
Step @("-m", "src.experiments.export_artifacts")
Step @("-m", "pytest", "-q")
