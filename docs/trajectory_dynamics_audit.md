# Trajectory Dynamics Audit (Audit 6)

> Diagnostic analysis only. Does **not** modify `models/best.pt`, the production Kalman filter,
> `src/tracking_pipeline.py`, training, labels, or Streamlit.

## 1. Objective

Study the temporal dynamics of the detected trajectory on the single 130-frame Photron shot,
compare classical-CV reference motion with YOLO detections, and relate MISS frames to
apparent speed/acceleration — without claiming independent ground-truth accuracy.

## 2. Data

- Frames directory: `local Photron frames dir (via --frames-dir)`
- Frames processed: `130`
- Model: `models/best.pt` (read-only)
- CV source: background subtraction + contour centroid (pseudo-label / **reference labels**, not independent GT)
- Coordinates: **image pixels** (not metric)

## 3. Methodology

1. Recompute CV reference centers with the documented pseudo-label algorithm (threshold=30, min area>5, fixed 20×20 box).
2. Run YOLO `best.pt` at conf=0.25 (diagnostic threshold from prior model audit).
3. Build per-frame series: CV (x,y), YOLO (x,y,conf), deltas, finite-difference velocity/acceleration.
4. Fit linear `y = y0 + v t` and quadratic `y = y0 + v t + 0.5 a t²` on CV Y(t) (and report YOLO fits).
5. Overlay MISS regions (0–18, mid, 88–129) against local dynamics.

No production Kalman parameters were changed. No separate production-filter state is required for this diagnostic;
kinematics below use finite differences on measured centers.

## 4. Results — detection coverage

- CV-positive frames: `83` / 130
- YOLO DETECTED @0.25: `71`
- YOLO MISS @0.25: `59`

### Regional miss rates

| Region | Frames | YOLO MISS | Miss rate | Mean CV speed (px/frame) | Mean |a_y| |
|---|---:|---:|---:|---:|---:|
| early_0_18 | 19 | 19 | 1.000 | 6.51277424558885 | None |
| mid_19_87 | 69 | 0 | 0.000 | 6.953528274032359 | 0.2525180974538979 |
| late_88_129 | 42 | 40 | 0.952 | 7.058869558687725 | 0.467042002095115 |

## 5. Linear vs quadratic (CV reference Y)

- Linear: R² = `0.999729`, RMSE = `2.8981` px, v = `-7.1082` px/frame
- Quadratic: R² = `0.999850`, RMSE = `2.1568` px, v = `-6.7120` px/frame
- Apparent acceleration a (image): `-0.006638` px/frame²
- ΔR² (quad − lin) = `0.000121`
- ΔRMSE (lin − quad) = `0.7413` (positive ⇒ quadratic lower error)

Interpretation (cautious): any quadratic coefficient is **apparent acceleration in pixel space**.
It is **not** physical m/s² without camera calibration and geometry.

### YOLO series fits (detected frames only)

- Points: `71`
- Linear Y R²=`0.999866`, RMSE=`1.7198`
- Quadratic Y R²=`0.999940`, RMSE=`1.1546`, a=`-0.005888`

## 6. MISS vs dynamics

- Mean CV speed on MISS frames (where CV exists): mean≈7.11 px/frame (n=8)
- Mean CV speed on DETECTED frames: mean≈6.94 px/frame (n=71)
- Mean |a_y| on MISS: ≈0.54 (n=6) vs DETECTED ≈0.25 (n=70)
- Near-border MISS count: `3` / Near-border DETECTED: `1`

Evidence is descriptive only — correlation with speed/acceleration/border is measured,
not asserted as the sole causal mechanism.

## 7. Residuals

Largest |quadratic residuals| on CV Y cluster near frames ~88–106 (late phase onset),
e.g. frame 106 (~7.3 px), frame 90 (~5.7 px).

## 8. Limitations

- Single continuous shot (130 frames); no multi-shot generalization.
- CV labels are automatic reference labels/auto-labels, **not** independent ground truth.
- Train/val leakage risk remains for the archived YOLO weights.
- Units are pixels and frames, not meters or seconds of physical flight.
- Apparent acceleration must not be interpreted as gravity without calibration.
- Production Kalman was not modified and is not required for this finite-difference diagnostic.

## 9. Conclusion

YOLO detections concentrate in the mid-shot window (frames 19–87) with zero misses there.
Early misses align with absent/weak CV signal; late misses coincide with sparser CV positives,
slightly higher |a_y|, and occasional border proximity. A global quadratic improves RMSE modestly
(~0.74 px) over linear on CV Y, with very small apparent acceleration in pixel units.

This audit identifies **where** behavior degrades along the sequence; it does not validate absolute
accuracy or prescribe model changes.

Generated (UTC): `2026-09-06T02:55:24.686233+00:00`
