# Reproducibility

## What can be reproduced from this repository

| Step | Reproducible? | Notes |
|------|---------------|-------|
| Load official `ballistic.hdf5` | Yes (external download) | SHA256 provided in [dataset.md](dataset.md) |
| Extract the 130 Photron frames of shot `7.62x51mm/0` | Yes | Code pattern documented in [data_pipeline.md](data_pipeline.md) |
| Regenerate the same style of pseudo-labels | Yes | Exact algorithm & parameters in [pseudo_labeling.md](pseudo_labeling.md) |
| Retrain YOLOv8n under the same protocol | Yes | Configuration in [yolo_training.md](yolo_training.md) |
| Run the tracking pipeline on the extracted frames | Yes | `src/tracking_pipeline.py` + `models/best.pt` |
| Obtain identical numeric metrics | No guarantee | Depends on exact random seeds, Ultralytics version, and hardware |

## Required external resources

1. `ballistic.hdf5` (official DVS Benchmark 2021 release)
2. Python environment with:
   - `ultralytics`
   - `opencv-python`
   - `h5py`
   - `numpy`
   - `torch` (CPU or GPU)

See `requirements.txt` at the repository root for the versions used in the original experiment.

## Recommended reproduction outline

```text
1. Download ballistic.hdf5 and verify SHA256.
2. Extract frames:
     ballistic/7.62x51mm/0/photron/image_sequence → 130 PNGs
3. Run the pseudo-labeling algorithm (background = frame_000,
   threshold = 30, min area = 5, fixed box 20×20).
4. Build dataset.yaml with train = val = images/train.
5. Train:
     YOLO("yolov8n.pt").train(data=..., epochs=15, imgsz=640)
6. Run tracking_pipeline.py on the extracted frames
   using the newly produced (or the archived) best.pt.
```

## Known non-reproducibility factors

- Pseudo-label generation is deterministic given the same frames and parameters.
- YOLO training involves stochastic data augmentation and optimizer behavior; exact weight values may differ.
- The original Colab runtime used a specific Ultralytics / PyTorch combination; version drift can change numeric results.
- No random-seed locking was applied beyond `SEED = 42` for NumPy in the notebook setup cell.

## Integrity of the archived model

The file `models/best.pt` shipped in this repository is the exact checkpoint produced by the 15-epoch run described in the documentation. It should be treated as a historical artifact of that experiment, not as a model trained on official ground-truth labels.
