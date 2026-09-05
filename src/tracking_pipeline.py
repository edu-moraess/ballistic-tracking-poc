import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def run_pipeline(model_path, frames_dir, output_video_path, fps=10):
    model = YOLO(str(model_path))
    frames_paths = sorted(list(Path(frames_dir).glob("*.png")))
    if not frames_paths:
        raise FileNotFoundError(f"Nenhum frame encontrado em: {frames_dir}")

    first_frame = cv2.imread(str(frames_paths[0]))
    h, w, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    trajectory_points = []

    for idx, path in enumerate(frames_paths):
        frame = cv2.imread(str(path))
        results = model(frame, verbose=False, conf=0.25)[0]
        center_x, center_y = None, None

        if len(results.boxes) > 0:
            box = results.boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 127), 2)
            cv2.putText(frame, f"Projetil {conf:.2f}", (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 127), 1)

        if center_x is not None and center_y is not None:
            meas = np.array([[np.float32(center_x)], [np.float32(center_y)]])
            kf.correct(meas)

        pred = kf.predict()
        px, py = int(pred[0, 0]), int(pred[1, 0])

        if center_x is not None:
            trajectory_points.append((px, py))

        for i in range(1, len(trajectory_points)):
            cv2.line(frame, trajectory_points[i - 1], trajectory_points[i], (0, 0, 255), 2)

        cv2.putText(frame, f"Frame: {idx:03d}/{len(frames_paths)} | {fps} FPS", 
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        out.write(frame)

    out.release()

if __name__ == "__main__":
    run_pipeline("models/best.pt", "data/extracted/photron", "docs/demo_tracking.mp4", fps=10)
