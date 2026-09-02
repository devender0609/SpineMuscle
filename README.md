# SpineMuscle AI v15 — Integrated Research Pipeline

This package is the first app-level integration of the workflow that was successfully demonstrated on patient 0271.

## What the UI now supports
1. One-study DICOM upload.
2. Frozen-v3 level proposal stage.
3. A single four-level confirmation screen.
4. Automated downstream processing.
5. A research report screen with:
   - four-level muscle SI numerator
   - CSF SI when frozen v5.2 can legitimately issue it
   - PVMQ when accepted
   - QC tier / decision
   - level-specific CSA and signal measurements
   - explicit research-only interpretation boundary

The frontend is designed for Vercel. The inference service is designed for a persistent Docker host such as Render/Railway, not Vercel serverless.

## Critical scientific limitation carried forward
The proven frozen PVMQ AUTO v5.2 method uses a **dataset-specific thecal-sac working mask** as part of its CSF hierarchy.

Patient 0271 could be processed end-to-end because that compatible mask existed in the research dataset.

For an arbitrary new uploaded MRI, the app MUST NOT invent or substitute a CSF ROI. Until a separately validated automatic thecal-sac mask generator is installed, the backend must return `PVMQ_NOT_ISSUED` for new cases lacking the required frozen-v5.2 mask input.

That is intentional governance, not an app bug.

## Frozen assets that must be installed on the inference host
Do not commit large model files to the public frontend repository.

Required:
- `frozen/level_v3/best_model.pt` — the frozen v3 checkpoint
- exact v1.5 LevelVerify inference implementation ported into `frozen/level_v3/generate_level_review_api.py`
- official MuscleMap v1.4 environment / weights
- exact frozen `run_pvmq_v5_2.py`
- the exact confirmed-plane geometry + muscle metric implementation already validated
- compatible research CSF/thecal-sac masks when processing the existing dataset

The two Python files currently supplied as **contracts** deliberately fail closed until the byte-faithful validated local code is installed. This avoids silently reconstructing a frozen algorithm from memory.

## Frontend local run
```powershell
cd frontend
npm install
npm run dev
```

Create `frontend/.env.local`:
```text
VITE_API_URL=http://localhost:8080
```

## Inference API local run
The API shell can be started after installing its dependencies:
```powershell
cd inference
python -m pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080
```

The health endpoint will show whether the frozen assets are installed:
`/api/health`

## Deployment
### Vercel frontend
Use `frontend` as the Vercel Root Directory.
- Framework: Vite
- Build: `npm run build`
- Output: `dist`
- Env: `VITE_API_URL=https://<your inference service>`

### Inference service
Deploy the `inference` directory to a persistent container/GPU-capable service. For production research use, install the validated MuscleMap environment and frozen model assets in the image or a private volume.

## Safety / claim boundary
The application does not provide:
- a patient-specific PJK probability
- a validated clinical PVMQ cutoff
- diagnosis
- treatment advice
- scanner-independent claims

It is a technical research measurement workflow.
