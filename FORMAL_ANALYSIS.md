# SpineMuscle PVMQ v5.2 — Formal Technical Validation Analysis

## Analysis population
The final dataset contained 500 patients: 400 development and 100 prespecified locked-test patients. The frozen v5.2 algorithm was finalized before examining the locked-test results. Complete four-level muscle numerators were available for 390/400 development patients and 98/100 locked-test patients.

## Primary technical endpoint: automated PVMQ acceptance
Development: 229/390 accepted (58.7%; Wilson 95% CI 53.8%–63.5%).

Locked test: 68/98 accepted (69.4%; Wilson 95% CI 59.7%–77.6%).

The absolute locked-minus-development difference was 10.7% (Newcombe 95% CI -0.2% to 20.3%). Fisher exact p=0.064. Thus the locked-test acceptance rate was numerically higher but not clearly different at a two-sided 0.05 threshold.

## PVMQ distribution among accepted measurements
Development: mean 0.475 ± 0.218; median 0.412 [IQR 0.306–0.646], range 0.130–0.998.

Locked test: mean 0.452 ± 0.200; median 0.407 [IQR 0.318–0.559], range 0.168–0.986.

There was no evidence of a meaningful distribution shift: Welch t-test p=0.430; Mann–Whitney p=0.584; Kolmogorov–Smirnov p=0.593; Hedges g (locked minus development)=-0.10. No accepted PVMQ value exceeded 1 in either cohort.

## Scanner effects
Scanner-specific acceptance remained heterogeneous. In the locked test, acceptance was:
- GE 1.5T: 26/31 = 83.9%
- Philips 1.5T: 23/41 = 56.1%
- Philips 3T: 19/26 = 73.1%

The same directional pattern was present in development. In a logistic model pooling both cohorts and adjusting for cohort, Philips 1.5T had lower odds of acceptance than GE 1.5T (OR 0.31, 95% CI 0.20–0.48, p<2.79e-07). Philips 3T was not clearly different from GE 1.5T (OR 0.78, p=0.336).

Accepted PVMQ also differed across scanner groups (development Kruskal–Wallis p=0.0014; locked-test p=0.0116). Therefore scanner independence should not be claimed.

## QC tier stability
Accepted development cases: Tier A 130, Tier B 59, Tier C 40.
Accepted locked-test cases: Tier A 36, Tier B 14, Tier C 18.
The tier distribution did not show a clear cohort shift (chi-square p=0.236).

## Locked-test exclusions
Two of the 100 locked patients failed before PVMQ evaluation because a complete four-level muscle numerator could not be generated. Among the 98 entering frozen v5.2, 30 were excluded by the frozen QC hierarchy. The principal locked-test failure modes were unusable target segmentation, inability to obtain a defensible Tier A/B/C CSF reference, poorly separated GMM components, lack of a valid high-signal target candidate, and presumed CSF not brighter than muscle.

## Interpretation
The locked test supports **technical generalization of the frozen v5.2 measurement pipeline within this dataset**. The accepted PVMQ distribution was highly similar to development, the high-PVMQ failure tail did not recur, and the locked-test acceptance rate was not lower than development.

This is **not clinical validation of PJK prediction**. The current dataset does not establish a patient-specific PJK probability, a validated clinical cutoff, or scanner-independent performance. Any clinical-risk model would require linked clinical outcomes and a separate prespecified validation analysis.

## Recommended reporting language
“Following development and method freeze, the automated PVMQ pipeline was evaluated once in a prespecified 100-patient locked test cohort. Complete four-level muscle measurements were obtained in 98 patients, of whom 68 (69.4%) met the frozen automated PVMQ quality criteria. The accepted PVMQ distribution was similar to development (median 0.407 vs 0.412; Mann–Whitney p=0.584), and no accepted value exceeded 1 in either cohort. Scanner-specific differences in acceptance and PVMQ remained, particularly for Philips 1.5T, and should be considered a limitation.”
