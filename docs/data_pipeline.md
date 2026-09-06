# Data Pipeline

## Overview

```
ballistic.hdf5
      │
      ▼
Extraction of Photron frames
(ballistic/7.62x51mm/0/photron/image_sequence)
      │
      ▼
130 PNG frames (640×280, grayscale)
saved to extracted/photron/
      │
      ▼
Pseudo-label generation
(background subtraction + contour)
      │
      ▼
YOLO dataset layout
yolo_dataset/
├── images/train/   (130 PNGs)
├── labels/train/   (83 positive + 47 empty)
└── dataset.yaml
      │
      ▼
YOLOv8n training (15 epochs)
      │
      ▼
best.pt + tracking pipeline
```

## Frame extraction

Performed in the experiment notebook (`Yolotrain.ipynb`).

```python
with h5py.File(DATASET_PATH, 'r') as f:
    images = f['ballistic']['7.62x51mm']['0']['photron']['image_sequence'][()]

for idx, img in enumerate(images):
    cv2.imwrite(f"frame_{idx:03d}.png", img)
```

- Total frames extracted: **130**
- Naming convention: `frame_000.png` … `frame_129.png`
- Location (original experiment): `/content/drive/MyDrive/ballistic_tracking/extracted/photron/`

## YOLO dataset layout produced

```
yolo_dataset/
├── images/
│   └── train/
│       ├── frame_000.png
│       ├── ...
│       └── frame_129.png
├── labels/
│   └── train/
│       ├── frame_000.txt   # may be empty
│       ├── ...
│       └── frame_129.txt
└── dataset.yaml
```

## dataset.yaml used at training time

```yaml
path: /content/drive/MyDrive/ballistic_tracking/yolo_dataset
train: images/train
val: images/train          # same set as training
names:
  0: projetil
```

## Notes

- No temporal or shot-level split was performed.
- All 130 frames belong to a single continuous high-speed sequence.
- Frames and labels are **not** versioned in this GitHub repository (size / reproducibility policy).
- See [pseudo_labeling.md](pseudo_labeling.md) for how the `.txt` label files were created.
