"""
Integrated downstream contract.

This service MUST orchestrate the exact validated components already proven locally:
1) whole-study dcm2niix
2) MuscleMap v1.4 (NumPy 1.26.x compatible env)
3) geometry mapping of the four confirmed DICOM planes
4) four-level bilateral multifidus + erector-spinae mean-SI numerator
5) frozen PVMQ AUTO v5.2 only when the exact required dataset-specific thecal-sac
   mask input is available and passes its Tier A/B/C hierarchy.

For arbitrary new scans where that frozen v5.2 mask input does not exist, return:
  pvmq_issued=false
  qc_decision="PVMQ_NOT_ISSUED"
  qc_reason="Frozen v5.2 requires a compatible thecal-sac working mask; no validated automatic mask generator is installed."

Do not invent a CSF ROI, PVMQ ratio, clinical cutoff, or PJK probability.

Expected result.json keys:
{
 "muscle_numerator": 96.165,
 "csf_mean_si": 471.769 or null,
 "pvmq": 0.20384 or null,
 "pvmq_issued": true/false,
 "csf_tier": "Tier A" or null,
 "qc_decision": "AUTO_ACCEPT" / "AUTO_EXCLUDE" / "PVMQ_NOT_ISSUED",
 "qc_reason": "...",
 "levels":[{"level":"L1-L2","mean_si":...,"csa_mm2":...,"geometry_qc":"PASS"}, ...],
 "method_versions":{"level_localizer":"v3 frozen","musclemap":"v1.4","pvmq":"v5.2 frozen"}
}
"""
raise SystemExit("Install the exact validated local pipeline runner before production use.")
