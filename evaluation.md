Experimental Evaluation

1. Evaluation protocol

The experiment evaluates a visual tracking pipeline composed of:

1. Classical computer vision measurements;
2. YOLO-based object detection;
3. Kalman-filter state estimation;
4. One-step-ahead state prediction.

The evaluated sequence contains 130 Photron frames with a resolution of 640 × 280 pixels.

The classical computer vision detector is based on background subtraction and contour analysis. Because these measurements were also used to generate the pseudo-labels used during YOLO training, they are treated as a reference signal, not as independent ground truth.

Therefore, the reported errors should be interpreted as agreement with the classical CV reference rather than as absolute detection accuracy.

---

2. Audit 1 — YOLO × Classical CV

The first audit compared YOLO detections against the classical CV reference using a confidence threshold of 0.25.

Results:

Metric| Result
Total frames| 130
CV-positive frames| 83
YOLO-positive frames| 71
Mean center error| 0.576 px
Median center error| 0.334 px
Maximum center error| 6.288 px

The low median displacement indicates that, on frames where both methods detected the object, YOLO produced bounding-box centers very close to the classical CV measurements.

However, this result does not constitute an independent accuracy measurement because the YOLO training labels originated from the same classical CV pipeline.

---

3. Audit 2 — Low-confidence detections

The 12 CV-positive frames not detected by YOLO at confidence ≥ 0.25 were re-evaluated using a very low inference threshold.

All 12 frames produced a YOLO response below the original confidence threshold.

This indicates that the apparent misses at confidence ≥ 0.25 were primarily low-confidence detections rather than complete absence of model response.

The recovered detections had a mean confidence of approximately 0.13.

This experiment was used diagnostically and was not considered a valid detection threshold for the production pipeline.

---

4. Audit 3 — Ungated Kalman experiment

A preliminary Kalman experiment allowed YOLO detections with confidence as low as 0.001 to enter the filter.

This produced large transient errors during initialization, with the largest Kalman-to-CV displacement reaching approximately 41 px.

The result demonstrated that low-confidence detector responses can destabilize state estimation, motivating the use of measurement gating.

This experiment therefore served as a diagnostic experiment rather than a final performance measurement.

---

5. Audit 4 — Gated Kalman estimation

The Kalman filter was then evaluated using only YOLO detections with confidence ≥ 0.25.

Results:

Comparison| Mean| Median| Maximum
YOLO × CV| 0.576 px| 0.334 px| 6.288 px
Kalman × CV| 1.492 px| 0.315 px| 16.540 px
Kalman × YOLO| 0.321 px| 0.239 px| 2.181 px

The Kalman estimate achieved a median displacement error of 0.315 px relative to the CV reference.

The larger mean error was primarily associated with frames in which YOLO did not provide a valid measurement and the filter therefore operated in prediction-only mode.

The estimated vertical velocity converged to approximately -7 px/frame during the stable portion of the sequence.

These quantities are expressed in image coordinates and should not be interpreted as physical projectile velocity.

---

6. Audit 5 — One-step-ahead prediction

The fifth audit evaluated the Kalman prediction before measurement correction.

Results:

Metric| Result
Prediction × CV mean error| 1.691 px
Prediction × CV median error| 0.365 px
Prediction × CV maximum error| 16.540 px
Gap-frame mean error| 7.731 px
Gap-frame median error| 8.283 px
Gap-frame maximum error| 16.540 px

The median one-step prediction error was 0.365 px relative to the classical CV reference.

During periods with valid observations, the prediction converged rapidly toward the observed trajectory.

When the detector produced gaps, prediction error increased progressively because the constant-velocity model was propagated without measurement correction.

---

7. Interpretation

The experiments indicate that the current pipeline can maintain a highly consistent image-space trajectory estimate on the evaluated sequence.

The strongest observation is not the absolute error value itself, but the consistency between the three stages:

Measurement → State Estimation → Prediction

The detector provides measurements close to the CV reference, the Kalman filter smooths those measurements, and the state model provides short-horizon predictions.

At the same time, the experiments expose an important limitation: the current evaluation is based on a reference signal derived from the same CV methodology used to construct the YOLO pseudo-labels.

Consequently, these results demonstrate internal consistency and pipeline behavior, not independent generalization performance.

---

8. Current limitations

The current experiment has several methodological limitations:

- The dataset contains a single evaluated shot from the 7.62×51 mm class.
- YOLO training and validation use the same image sequence.
- The training annotations are pseudo-labels rather than manually verified ground truth.
- The classical CV reference is therefore not independent of the training process.
- The Kalman model operates in 2D image coordinates.
- The current state transition model assumes constant velocity.
- No camera calibration or physical coordinate reconstruction is used in the current tracking evaluation.
- The reported velocity is expressed in pixels per frame.
- Long prediction gaps accumulate error.

These limitations are explicitly retained as part of the experimental record rather than hidden from the evaluation.

---

9. Next experiment

The next experiment will evaluate whether the observed trajectory is adequately represented by the constant-velocity model.

A linear trajectory model will be compared against a second-order model using the observed CV trajectory.

The comparison will quantify whether systematic acceleration or curvature remains in the residuals.

This experiment will determine whether the current constant-velocity state model is sufficient for the evaluated sequence.