"""
API adapter contract for the already-frozen v3 localizer.

IMPORTANT:
Do not retrain or change best_model.pt.
Port the proposal-generation portion of the validated LevelVerify v1.5 code here,
preserving preprocessing and score calculation exactly.

Required output:
  <output-dir>/level_proposals.json
  <output-dir>/previews/*.jpg

JSON schema:
{
  "L1-L2": [{"dicom_file":"IM000023.dcm","score":0.9999,"preview_id":"L1_L2_1.jpg"}, ... top 5],
  "L2-L3": [...],
  "L3-L4": [...],
  "L4-L5": [...]
}

This file is intentionally a contract, not a reconstructed model implementation.
Copy the exact inference body from the validated v1.5 LevelVerify generator on the
deployment machine so the frozen model semantics are not silently altered.
"""
raise SystemExit("Install the exact validated v1.5 LevelVerify inference body before production use.")
