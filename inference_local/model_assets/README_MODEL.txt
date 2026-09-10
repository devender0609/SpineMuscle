SpineMuscle v15.1.3.5 CPU + auto-cleanup

Before git add/push, copy:
  C:\Users\dpsingh\Desktop\musclemap_v1_4_model.zip
to:
  inference_local\model_assets\musclemap_v1_4_model.zip

Expected ZIP SHA-256:
CF0E86B7328A83D31AA79FC9514622949C6A935DE5E7C6A36D3C5F14DF2CFC5D

Extracted expected hashes:
PTH  45DFA2843D2E0B1FD842152D6A79BFD4BCB90899C076CEB6346C59DA9A79A16C
JSON 82C74F854AB74D8770E6E2B9A240BD23CCC646A7B71B13E6CFD8F9919D2ACB26

Runtime changes:
- CPU-only PyTorch wheels in backend and MuscleMap environments
- only one heavy MuscleMap analysis executes at a time
- app-owned subprocess output streams to disk instead of RAM
- DICOM/NIfTI/segmentation/intermediate folders purge after terminal result
- small report/job/log metadata expire after 24 hours by default
- frozen exact analysis scripts and frozen level checkpoint are unchanged
