# Render deployment — Docker runtime (v15.1.3)

The cloud backend now uses Docker because the frozen MuscleMap v4 runner requires a
separate Conda environment named `MuscleMap`, the `dcm2niix` executable, and the
`mm_segment` console command.

## Why Docker

The frozen file `exact/musclemap_v4/run_musclemap_v4.py` is intentionally unchanged.
It calls:

- `conda run -n MuscleMap python ...`
- `conda run -n MuscleMap dcm2niix ...`
- `conda run -n MuscleMap mm_segment ...`

The Docker image reproduces that contract instead of rewriting the frozen method.

## Render settings

Create/deploy the service from the repository with:

- **Runtime:** Docker
- **Root Directory:** `inference_local`
- **Dockerfile:** `Dockerfile`
- **Health check:** `/api/health`
- **Recommended memory:** at least 2 GB; 4 GB if a full study still approaches the limit

Environment variables:

- `CORS_ORIGINS=https://spine-muscle.vercel.app`
- `SPINEMUSCLE_WORKDIR=/var/data/work`
- `SPINEMUSCLE_HOST_PIPELINE=1`

Attach the persistent disk:

- **Mount path:** `/var/data`
- **Size:** 10 GB

The image defines `HF_HOME=/var/data/huggingface` and `TORCH_HOME=/var/data/torch` so
runtime model/cache downloads can persist on the Render disk.

## Two pinned Python environments

`spinemuscle-backend` (Python 3.12.11) preserves the validated web/level-localizer
stack, including NumPy 2.2.6, SciPy 1.13.1, torch 2.13.0, and torchvision 0.28.0.

`MuscleMap` (Python 3.11.14) reproduces the observed local runtime, including NumPy
1.26.4, MONAI 1.3.2, torch 2.13.0, torchvision 0.28.0, `scripts==2.0` (which supplies
`scripts.mm_segment:main`), and `dcm2niix=1.0.20260416`.

## Important validation boundary

The Docker image reproduces the observed software versions and command contract, but
cloud execution of MuscleMap is not considered validated until a known reference case
completes end-to-end and its outputs are compared with the validated local result.
Do not change frozen method files to make a cloud run pass.

## Vercel

Keep:

`VITE_API_URL=https://spinemuscle-api.onrender.com`

Redeploy Vercel only after the Docker backend is healthy.


## MuscleMap source pin (v15.1.3.3)
The Docker runtime installs MuscleMap from the exact clean source commit validated locally: `6e1e1eb6732337c13cab53bd5cc800c69024774f` from `https://github.com/MuscleMap/MuscleMap.git`. It does not use the unrelated PyPI `scripts==2.0` distribution. The installation is `--no-deps`; the environment file declares the validated runtime dependencies explicitly.


## v15.1.3.4 bundled model
Copy `musclemap_v1_4_model.zip` into `inference_local/model_assets/` before pushing. Docker verifies all supplied SHA-256 hashes and seeds MuscleMap wholebody v1.4 locally.

## v15.1.3.5 CPU / memory / retention profile

Recommended Render environment:
- `SPINEMUSCLE_WORKDIR=/var/data/work`
- `SPINEMUSCLE_JOB_RETENTION_HOURS=24`
- `SPINEMUSCLE_PURGE_HEAVY_ON_TERMINAL=1`
- `CORS_ORIGINS=https://spine-muscle.vercel.app`

This build attempts to keep the existing 2 GB service feasible by using CPU-only
PyTorch wheels, serializing heavy jobs, streaming app-owned subprocess logs, and
automatically removing bulky study data after terminal results. It does not alter
the frozen exact analysis scripts or checkpoint.

After deployment, rerun 0271 first and require exact reference reproduction before
accepting this runtime optimization as the new baseline.
