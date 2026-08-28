# SpineMuscle AI Research Dashboard

Deployable Next.js dashboard for the frozen PVMQ v5.2 technical-validation results.

## GitHub
Create a new GitHub repository and push this folder.

## Vercel
1. Import the GitHub repository into Vercel.
2. Framework preset: Next.js.
3. No environment variables are required for this validation-dashboard release.
4. Deploy.

## Important architecture note
This Vercel release intentionally does **not** run MuscleMap, dcm2niix, or arbitrary DICOM inference.

The validated 500-patient pipeline relied on exact lumbar-level mapping from dataset-specific published annotations. That level-localization method does not generalize automatically to new MRI studies.

For one-upload MRI analysis, the next production phase should add:
- a separately hosted containerized inference backend with dcm2niix + MuscleMap;
- a generalizable, validated lumbar-level localization component;
- PHI/security controls appropriate to the deployment environment;
- then connect the Vercel frontend to that API.

Do not market this build as clinically validated PJK prediction.
