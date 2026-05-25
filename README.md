# Rozendal Thesis Pipeline
### Pre-ictal Pupil Dynamics as Predictors of Absence Seizures
**Replicates Nestvogel & McCormick (2022) Figure 5F + pre-ictal pupil analysis**

---

## Setup

1. Download all 7 files and put them in ONE folder on your Desktop:
   ```
   ~/Desktop/rozendal_pipeline/
       SETUP_AND_RUN.sh
       run_all_recordings.py
       validate_pipeline.py
       load_data.py
       thalamic_burst_detection.py
       event_alignment.py
       v1_processing.py
   ```

2. Plug in the hard drive (`/Volumes/home/Dennis_to_Kevin/LP/`)

3. Open Terminal and run:
   ```bash
   cd ~/Desktop/rozendal_pipeline
   bash SETUP_AND_RUN.sh
   ```

That's it. The script installs Python packages, runs the pipeline, then runs validation.

---

## What each file does

| File | Purpose | You run this? |
|------|---------|---------------|
| `SETUP_AND_RUN.sh` | Installs packages, runs pipeline + validation | **YES — run this** |
| `run_all_recordings.py` | Main pipeline: all 3 recordings → figures | Called by SETUP_AND_RUN.sh |
| `validate_pipeline.py` | 3-check validation (permutation, PETH benchmark, raw inspection) | Called by SETUP_AND_RUN.sh |
| `load_data.py` | Loads .mat files (handles both old and v7.3 HDF5 format) | Never run directly |
| `thalamic_burst_detection.py` | LP burst detection (faithful port of Dennis's MATLAB) | Never run directly |
| `event_alignment.py` | Aligns signals to oscillation onsets, pre-ictal analysis | Never run directly |
| `v1_processing.py` | Spike clipping (port of getspikes_and_clip_2021.m) | Never run directly |

---

## Output figures (saved to `output_figures/`)

**Per recording (DPV20_rec5, DPV21_rec6, RN51_rec3):**
- `{recording}_fig5F.png` — Figure 5F replication: pupil, whisking, walking, V1 Vm, FFT power aligned to LP oscillation onset
- `{recording}_preictal_pupil.png` — Pre-ictal pupil: mean trace, slope distribution, per-event Δpupil

**Grand average:**
- `grand_average_preictal_pupil.png` — Pooled across all recordings (primary thesis figure)

**Validation (in `output_figures/DPV20_rec5/`):**
- `check1_permutation.png` — Shuffled null vs. real trace (p-value)
- `check2_dennis_peth_benchmark.png` — Our alignment vs. Dennis's pre-computed PETH
- `check3_raw_inspection.png` — Raw trace + individual trials + heatmap

---

## Data paths (edit in run_all_recordings.py if needed)
```
/Volumes/home/Dennis_to_Kevin/LP/DPV20_rec5_analyzed_wspikes_NRPX_sync_corrected.mat
/Volumes/home/Dennis_to_Kevin/LP/DPV21_rec6_analyzed_wspikes_and_NRPX_sync_corrected.mat
/Volumes/home/Dennis_to_Kevin/LP/RN51_rec3_L23_LP_NRPX_sync_corrected.mat
```
