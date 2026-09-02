# SpineMuscle AI v15.1 — Backend-Complete Research Integration

v15.1 replaces the placeholder backend with a working **local inference bridge** that calls the exact validated components already proven on the Windows workstation.

## Wired in
- exact LevelVerify v1.5 inference code
- frozen v3 checkpoint hash gate
- exact MuscleMap v4 runner
- exact confirmed-plane mapping v5 runner
- exact frozen PVMQ AUTO v5.2
- 0271 reproducibility masks/manifest/level mapping
- FastAPI endpoints used by the Vercel frontend
- optional Research dataset case ID
- clear offline-backend error messages

## Run order
1. Push v15.1 to GitHub; let Vercel redeploy `frontend`.
2. `cd inference_local`
3. `powershell -ExecutionPolicy Bypass -File .\STAGE_AND_VERIFY_ASSETS.ps1`
4. In PowerShell window 1: `powershell -ExecutionPolicy Bypass -File .\START_BACKEND.ps1`
5. Confirm `http://127.0.0.1:8080/api/health` returns `"ok": true`.
6. In PowerShell window 2: `powershell -ExecutionPolicy Bypass -File .\START_TEMP_HTTPS_TUNNEL.ps1`
7. Copy the generated `https://...trycloudflare.com` URL.
8. In Vercel set `VITE_API_URL` to that URL and redeploy.
9. Upload patient 0271 and enter `0271` in Research dataset case ID.
10. Confirm the four planes and verify the app reproduces approximately PVMQ 0.20384 / Tier A / AUTO_ACCEPT.

For a new study without a compatible validated thecal-sac mask, the app will complete muscle analysis but deliberately return `PVMQ_NOT_ISSUED`.

No PJK probability, clinical cutoff, diagnosis, or treatment recommendation is produced.


## v15.1.1 fixes
- `STAGE_AND_VERIFY_ASSETS.ps1` now creates `inference_local\frozen\level_v3` before copying `best_model.pt`.
- `.keep` preserves the otherwise-empty checkpoint folder through ZIP extraction.
- `scripts\SAFE_PUSH_TO_SPINEMUSCLE.ps1` refuses to push when `origin` is not exactly `https://github.com/devender0609/SpineMuscle.git`.
