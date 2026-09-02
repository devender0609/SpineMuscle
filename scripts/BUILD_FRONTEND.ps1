Set-Location "$PSScriptRoot\..\frontend"
npm install
npm run build
if($LASTEXITCODE -ne 0){throw "Frontend build failed"}
Write-Host "Frontend build successful" -ForegroundColor Green
