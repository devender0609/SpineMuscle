$ErrorActionPreference="Stop"
$expected="https://github.com/devender0609/SpineMuscle.git"
Write-Host "Checking Git repository..." -ForegroundColor Cyan

$top=(git rev-parse --show-toplevel 2>$null)
if($LASTEXITCODE -ne 0){throw "This folder is not inside a Git repository."}

$remote=(git remote get-url origin 2>$null)
if($LASTEXITCODE -ne 0){throw "No origin remote is configured."}

Write-Host "Git root: $top"
Write-Host "Origin:   $remote"

if($remote -ne $expected){
  throw "STOP: origin is not the SpineMuscle repository. Expected $expected"
}

git status
Write-Host ""
Write-Host "Remote is correct. Safe to push SpineMuscle changes." -ForegroundColor Green
