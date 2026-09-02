# v15 integration status

## Proven and represented in the app
- single-study upload flow
- one-screen four-level review UX
- async job/status API
- report UX
- strict research-only labeling
- failure-closed PVMQ behavior
- Vercel frontend / container inference architecture

## Must be copied from the validated local workstation before production deployment
- exact frozen-v3 LevelVerify inference body
- frozen v3 `best_model.pt`
- exact working MuscleMap v1.4 runtime/weights
- exact confirmed-plane geometry + metric code
- exact frozen PVMQ AUTO v5.2 implementation
- dataset-specific CSF/thecal-sac masks where applicable

These are intentionally not reimplemented from memory in this ZIP.
