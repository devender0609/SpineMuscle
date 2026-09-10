FROM continuumio/miniconda3:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SPINEMUSCLE_HOST_PIPELINE=1 \
    SPINEMUSCLE_WORKDIR=/var/data/work \
    SPINEMUSCLE_JOB_RETENTION_HOURS=24 \
    SPINEMUSCLE_PURGE_HEAVY_ON_TERMINAL=1 \
    HF_HOME=/var/data/huggingface \
    TORCH_HOME=/var/data/torch

WORKDIR /app

COPY backend-environment.yml /tmp/backend-environment.yml
COPY musclemap-environment.yml /tmp/musclemap-environment.yml
COPY musclemap-cpu-requirements.txt /tmp/musclemap-cpu-requirements.txt
COPY model_assets/musclemap_v1_4_model.zip /tmp/musclemap_v1_4_model.zip

# Create both runtimes.
RUN conda env create -f /tmp/backend-environment.yml \
    && conda env create -f /tmp/musclemap-environment.yml

# Install CPU-only PyTorch wheels explicitly. This prevents CUDA/NVIDIA runtime
# packages from being pulled into a CPU Render deployment.
RUN /opt/conda/envs/spinemuscle-backend/bin/python -m pip install \
       torch==2.13.0 torchvision==0.28.0 \
       --index-url https://download.pytorch.org/whl/cpu \
    && /opt/conda/envs/MuscleMap/bin/python -m pip install \
       torch==2.13.0 torchvision==0.28.0 \
       --index-url https://download.pytorch.org/whl/cpu

# Install the validated MuscleMap support stack after CPU torch is already present,
# then install the exact validated MuscleMap source commit without replacing deps.
RUN /opt/conda/envs/MuscleMap/bin/python -m pip install -r /tmp/musclemap-cpu-requirements.txt \
    && /opt/conda/envs/MuscleMap/bin/python -m pip install --no-deps \
       "git+https://github.com/MuscleMap/MuscleMap.git@6e1e1eb6732337c13cab53bd5cc800c69024774f"

# Seed the exact validated wholebody v1.4 model locally so runtime never depends on Zenodo.
RUN /opt/conda/envs/MuscleMap/bin/python - <<'PY'
from pathlib import Path
import hashlib, zipfile
import scripts.mm_segment as mm

zip_path = Path("/tmp/musclemap_v1_4_model.zip")
expected_zip = "CF0E86B7328A83D31AA79FC9514622949C6A935DE5E7C6A36D3C5F14DF2CFC5D"
expected_pth = "45DFA2843D2E0B1FD842152D6A79BFD4BCB90899C076CEB6346C59DA9A79A16C"
expected_json = "82C74F854AB74D8770E6E2B9A240BD23CCC646A7B71B13E6CFD8F9919D2ACB26"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()

assert sha256(zip_path) == expected_zip, "Bundled model ZIP hash mismatch"

scripts_dir = Path(mm.__file__).resolve().parent
target = scripts_dir / "models" / "wholebody" / "v1.4"
target.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(target)

pth = target / "contrast_agnostic_wholebody_model.pth"
js = target / "contrast_agnostic_wholebody_model.json"
assert pth.exists() and js.exists()
assert sha256(pth) == expected_pth, "MuscleMap v1.4 .pth hash mismatch"
assert sha256(js) == expected_json, "MuscleMap v1.4 .json hash mismatch"
print("Bundled MuscleMap wholebody v1.4 model verified:", target)
PY

RUN conda clean -afy
COPY . /app

# Build-time runtime checks.
RUN /opt/conda/envs/spinemuscle-backend/bin/python -c \
       "import torch,torchvision,numpy,scipy,pydicom,PIL; assert torch.version.cuda is None, torch.version.cuda; print('backend CPU runtime ok', torch.__version__)" \
    && conda run -n MuscleMap python -c \
       "import numpy,monai,torch; assert numpy.__version__=='1.26.4'; assert monai.__version__=='1.3.2'; assert torch.version.cuda is None, torch.version.cuda; print('MuscleMap CPU runtime ok', torch.__version__)" \
    && test -x /opt/conda/envs/MuscleMap/bin/dcm2niix \
    && (/opt/conda/envs/MuscleMap/bin/dcm2niix --version || [ $? -eq 3 ]) \
    && conda run -n MuscleMap python -c \
       "import scripts.mm_segment as m; assert callable(getattr(m,'main',None)); print('scripts.mm_segment module ok:', m.__file__)" \
    && test -x /opt/conda/envs/MuscleMap/bin/mm_segment \
    && conda run -n MuscleMap python -c \
       "from pathlib import Path; import scripts.mm_segment as m; p=Path(m.__file__).resolve().parent/'models'/'wholebody'/'v1.4'/'contrast_agnostic_wholebody_model.pth'; assert p.exists() and p.stat().st_size==104884148; print('MuscleMap v1.4 model present:', p)"

EXPOSE 10000
CMD ["/bin/bash", "-lc", "/opt/conda/envs/spinemuscle-backend/bin/python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
