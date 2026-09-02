$ErrorActionPreference="Stop"
Write-Host "SpineMuscle AI v15.1 - stage and verify validated local assets" -ForegroundColor Cyan
$expected="2EB1767910D1934F6CEA4152E09CADE95847ECE5777714BF63B9C1A9AFC125CC"
$src="$env:LOCALAPPDATA\SpineMuscleAI\LEVEL_V3_FROZEN\best_model.pt"
$dstDir="$PSScriptRoot\frozen\level_v3"
$dst="$dstDir\best_model.pt"
if(!(Test-Path $dstDir)){New-Item -ItemType Directory -Path $dstDir -Force | Out-Null}
if(!(Test-Path $src)){throw "Frozen v3 checkpoint not found at $src"}
$hash=(Get-FileHash $src -Algorithm SHA256).Hash
Write-Host "Frozen v3 SHA256: $hash"
if($hash -ne $expected){throw "Checkpoint hash mismatch. Expected $expected"}
Copy-Item $src $dst -Force
conda run -n SpineLevels python -c "import torch,pydicom,numpy,PIL; print('SpineLevels OK'); print('torch',torch.__version__); print('numpy',numpy.__version__)"
if($LASTEXITCODE -ne 0){throw "SpineLevels environment check failed"}
conda run -n MuscleMap python -c "import numpy,monai; print('NumPy',numpy.__version__); print('MONAI',monai.__version__); assert numpy.__version__.startswith('1.26.')"
if($LASTEXITCODE -ne 0){throw "MuscleMap environment is not in the validated NumPy 1.26.x state"}
conda run -n MuscleMap mm_segment --help | Out-Null
if($LASTEXITCODE -ne 0){throw "mm_segment unavailable"}
conda run -n MuscleMap dcm2niix -h | Out-Null
if($LASTEXITCODE -ne 0){throw "dcm2niix unavailable"}
Write-Host "All required assets/environment checks passed." -ForegroundColor Green
