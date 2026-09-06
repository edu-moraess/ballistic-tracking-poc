# Kalman Filter

## Role in the pipeline

After YOLOv8n produces a bounding box, the center of that box is treated as a 2D measurement and fed into a classic Kalman filter implemented with OpenCV. The filter produces a smoothed 2D image-space trajectory.

## State and measurement

| Quantity | Definition |
|----------|------------|
| State vector | `[x, y, vx, vy]ᵀ` |
| Measurement vector | `[x, y]ᵀ` |
| Motion model | Constant velocity |
| Implementation | `cv2.KalmanFilter(4, 2)` |

## Matrices used (as coded)

**Measurement matrix**
```
H = [[1, 0, 0, 0],
     [0, 1, 0, 0]]
```

**Transition matrix (constant velocity)**
```
F = [[1, 0, 1, 0],
     [0, 1, 0, 1],
     [0, 0, 1, 0],
     [0, 0, 0, 1]]
```

**Process noise covariance**
```
Q = 0.03 * I₄
```

Measurement noise covariance was left at the OpenCV default (not explicitly set in the experiment code).

## Update order in the current implementation

In the tracking script the order of operations is:

1. Run YOLO detection
2. If a detection exists → `kf.correct(measurement)`
3. Always → `kf.predict()`

The predicted state is used to draw the trajectory line.

## Scope

- Operates purely in **image coordinates** (pixels).
- No conversion to metric / world coordinates is performed.
- No use of camera calibration parameters from the HDF5.
- No 3D ballistic model (gravity, drag, etc.) is implemented.

See [trajectory.md](trajectory.md) for the resulting trajectory representation.
