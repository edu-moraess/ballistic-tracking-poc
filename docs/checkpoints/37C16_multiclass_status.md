# Checkpoint 37C.16 — Multiclass VisDrone Baseline

Date: 2026-09-06

## Scope

This checkpoint records the state of the multiclass computer-vision branch before restarting the runtime to enable GPU/CUDA training.

No production ballistic-control behavior is introduced by this checkpoint. The multiclass detector is a general object-detection experiment using VisDrone.

## Dataset

VisDrone DET was prepared as a YOLO dataset:

- Train images: 6,471
- Validation images: 548
- Total images: 7,019
- Train objects: 343,204
- Validation objects: 38,759
- Total objects: 381,963
- Classes: 10
- Image/label correspondence: 100%
- Invalid YOLO label lines: 0

Classes:

1. pedestrian
2. people
3. bicycle
4. car
5. van
6. truck
7. tricycle
8. awning-tricycle
9. bus
10. motor

The original VisDrone annotations remain preserved. Ignored regions and the official `others` category were excluded from the YOLO 10-class training set.

## Model construction

Base model:

- YOLO11n pretrained on COCO
- 80 classes

New model:

- YOLO11n architecture
- Real detection head with `nc=10`
- `Detect.nc = 10`
- `Detect.no = 74`
- 448 compatible parameter tensors transferred from the COCO model
- Output checkpoint: `yolo11n_visdrone_10cls_real.pt`
- Runtime checkpoint size: approximately 10.26 MB

The specialized historical `models/best.pt` detector remains separate and was not modified or used for this experiment.

## Pre-training inference audit — 37C.15

Eight validation images were tested before VisDrone training.

- Images tested: 8
- Raw detections at `conf=0.001`: 1,210
- Invalid class IDs: 0
- Confidence mean: 0.0013
- Confidence median: 0.0013
- Confidence minimum: 0.0013
- Confidence maximum: 0.0013

The very low and nearly constant confidence values, together with the unbalanced raw class responses, are expected from a newly reconstructed VisDrone detection head before task-specific training. This audit was structural rather than a performance evaluation.

## Training attempt — 37C.16

A pilot training run was started with:

- Model: `yolo11n_visdrone_10cls_real.pt`
- Dataset: VisDrone YOLO dataset
- Epochs requested: 10
- Image size: 640
- Batch: 8
- Device: CPU
- Workers: 2
- Seed: 42
- Separate run directory: `visdrone/runs/yolo11n_visdrone_pilot_37C16`

The run was stopped before producing persistent training artifacts.

Audit result:

- `results.csv`: absent
- `best.pt`: absent
- `last.pt`: absent
- Run directory: created
- Multiclass source checkpoint: preserved

Therefore, there are no pilot training metrics to interpret or claim as model performance.

## Next step

Restart the runtime with GPU/CUDA enabled, confirm the accelerator, and repeat the pilot training using CUDA. The CPU attempt is retained only as an execution-state checkpoint; it is not a benchmark result.
