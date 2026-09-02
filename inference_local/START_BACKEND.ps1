$ErrorActionPreference="Stop"
Write-Host "Starting SpineMuscle AI v15.1 local inference backend" -ForegroundColor Cyan
conda run -n SpineLevels python -m pip install -q -r "$PSScriptRoot\requirements.txt"
if($LASTEXITCODE -ne 0){throw "Backend dependency install failed"}
$env:SPINEMUSCLE_WORKDIR="$env:LOCALAPPDATA\SpineMuscleAI\WEB_PIPELINE_V15_1"
$env:CORS_ORIGINS="*"
Write-Host "Backend: http://127.0.0.1:8080" -ForegroundColor Green
Write-Host "Health:  http://127.0.0.1:8080/api/health" -ForegroundColor Green
Write-Host "Keep this window open." -ForegroundColor Yellow
conda run -n SpineLevels python -m uvicorn app:app --app-dir "$PSScriptRoot" --host 0.0.0.0 --port 8080
