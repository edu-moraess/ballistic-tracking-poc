# Trajectory Estimation

## What is produced

The current pipeline produces a **2D image-space trajectory**.

It is **not** a 3D ballistic trajectory, nor a metric-world trajectory.

## How it is built

1. YOLOv8n detects the projectile and returns a bounding box.
2. The center of the box `(cx, cy)` is extracted.
3. This center is passed as a measurement to the Kalman filter (see [kalman_filter.md](kalman_filter.md)).
4. The filter’s predicted state `(px, py)` is appended to a list of trajectory points.
5. Consecutive predicted points are connected by red line segments and written into the output video.

## Coordinate system

- Origin: top-left of the image
- Units: pixels
- Axes: x rightward, y downward (standard image convention)
- Resolution of the frames used: 640 × 280

## Visual artifacts

- `docs/plots/real_trajectory.png` – classical background-subtraction detections (red) overlaid with the Kalman estimate (lime) on a mid-sequence frame.
- `docs/demo_tracking.mp4` – full sequence with YOLO boxes (green) and Kalman trajectory (red), rendered at 10 FPS for visualization.

## Limitations

- Image-space only; no camera calibration or world-coordinate transform is applied.
- No physical ballistic model (gravity, air drag, spin, etc.).
- Trajectory continuity depends entirely on the quality and temporal density of the YOLO detections.
- When YOLO fails to detect the object, the filter coasts using the constant-velocity prediction only.
