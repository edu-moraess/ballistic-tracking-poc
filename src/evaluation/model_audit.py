"""Diagnostic frame-by-frame audit for the existing YOLO checkpoint.

This module only reads a model, image frames, and optional reference labels. It does
not train, alter weights, run the production tracker, or claim auto-labels are ground
truth. Run with::

    python -m src.evaluation.model_audit --frames-dir /path/to/frames
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Sequence

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_THRESHOLDS = (0.25, 0.50, 0.75)


@dataclass(frozen=True)
class Detection:
    detection_index: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    center_y: float
    width: float
    height: float


@dataclass(frozen=True)
class FrameRecord:
    threshold: float
    frame_id: int
    filename: str
    detection_count: int
    selected_detection_index: int | None
    confidence: float | None
    x1: float | None
    y1: float | None
    x2: float | None
    y2: float | None
    center_x: float | None
    center_y: float | None
    width: float | None
    height: float | None
    status: str
    detections: tuple[dict[str, Any], ...]


def natural_frame_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.stem)
    return (int(match.group(1)), path.name.lower()) if match else (10**12, path.name.lower())


def discover_frames(frames_dir: Path) -> list[Path]:
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory does not exist: {frames_dir}")
    frames = [path for path in frames_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(frames, key=natural_frame_key)


def _float(value: Any) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def parse_detections(result: Any) -> list[Detection]:
    """Convert one Ultralytics result into explicit detections, preserving all boxes."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = boxes.xyxy.cpu().tolist() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy.tolist()
    conf = boxes.conf.cpu().tolist() if hasattr(boxes.conf, "cpu") else boxes.conf.tolist()
    detections = []
    for index, (coords, confidence) in enumerate(zip(xyxy, conf)):
        x1, y1, x2, y2 = map(_float, coords[:4])
        detections.append(Detection(index, _float(confidence), x1, y1, x2, y2, (x1 + x2) / 2, (y1 + y2) / 2, max(0.0, x2 - x1), max(0.0, y2 - y1)))
    return detections


def select_detection(detections: Sequence[Detection]) -> Detection | None:
    return max(detections, key=lambda detection: (detection.confidence, -detection.detection_index), default=None)


def make_frame_record(threshold: float, frame_id: int, filename: str, detections: Sequence[Detection]) -> FrameRecord:
    selected = select_detection(detections)
    selected_dict = asdict(selected) if selected else None
    return FrameRecord(
        threshold=threshold, frame_id=frame_id, filename=filename,
        detection_count=len(detections),
        selected_detection_index=selected.detection_index if selected else None,
        confidence=selected.confidence if selected else None,
        x1=selected.x1 if selected else None, y1=selected.y1 if selected else None,
        x2=selected.x2 if selected else None, y2=selected.y2 if selected else None,
        center_x=selected.center_x if selected else None, center_y=selected.center_y if selected else None,
        width=selected.width if selected else None, height=selected.height if selected else None,
        status="DETECTED" if selected else "MISS",
        detections=tuple(asdict(detection) for detection in detections),
    )


def summarize(records: Sequence[FrameRecord], total_frames: int | None = None) -> dict[str, Any]:
    total = total_frames if total_frames is not None else len(records)
    detected = [record for record in records if record.status == "DETECTED"]
    confidences = [record.confidence for record in detected if record.confidence is not None]
    return {
        "total_frames": total,
        "frames_with_detection": len(detected),
        "frames_without_detection": total - len(detected),
        "detection_rate": len(detected) / total if total else None,
        "confidence_mean": mean(confidences) if confidences else None,
        "confidence_median": median(confidences) if confidences else None,
        "confidence_min": min(confidences) if confidences else None,
        "confidence_max": max(confidences) if confidences else None,
        "mean_detections_per_frame": mean([record.detection_count for record in records]) if records else 0.0,
        "missed_frames": [record.filename for record in records if record.status == "MISS"],
        "multiple_detection_frames": [record.filename for record in records if record.detection_count > 1],
    }


def _record_row(record: FrameRecord) -> dict[str, Any]:
    row = asdict(record)
    row["detections"] = json.dumps(row["detections"], separators=(",", ":"))
    return row


def write_records_csv(records: Iterable[FrameRecord], path: Path) -> None:
    rows = [_record_row(record) for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [field.name for field in FrameRecord.__dataclass_fields__.values()])
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_model(model_path: Path) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required to run inference; install the project's declared requirements") from exc
    return YOLO(str(model_path))


def run_audit(model_path: Path, frames: Sequence[Path], thresholds: Sequence[float]) -> tuple[list[FrameRecord], dict[str, Any]]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model does not exist: {model_path}")
    if not frames:
        raise ValueError("No image frames were found; refusing to run an audit on synthetic or substitute data")
    model = _load_model(model_path)
    records: list[FrameRecord] = []
    summary: dict[str, Any] = {"model_path": str(model_path), "frame_count": len(frames), "thresholds": [float(value) for value in thresholds], "by_threshold": {}}
    for threshold in thresholds:
        threshold_records = []
        for frame_id, frame_path in enumerate(frames):
            results = model.predict(source=str(frame_path), conf=float(threshold), verbose=False)
            detections = parse_detections(results[0])
            threshold_records.append(make_frame_record(float(threshold), frame_id, frame_path.name, detections))
        records.extend(threshold_records)
        summary["by_threshold"][str(threshold)] = summarize(threshold_records, total_frames=len(frames))
    return records, summary


def report_markdown(summary: dict[str, Any], frames_available: bool, frame_directory: str, annotation_source: str) -> str:
    lines = ["# YOLO Model Audit", "", "> Diagnostic audit only. It does not modify the model, pipeline, Kalman filter, Streamlit, or training artifacts.", "", "## Model", "", f"- Path: `{summary.get('model_path', 'models/best.pt')}`", "- Model type: YOLO checkpoint loaded by Ultralytics at runtime", f"- Confidence thresholds: {', '.join(map(str, summary.get('thresholds', DEFAULT_THRESHOLDS)))}", "", "## Dataset", "", f"- Frames directory: `{frame_directory}`", f"- Frames available: `{summary.get('frame_count', 0) if frames_available else 'NOT AVAILABLE'}`", "- Expected sequence: 130 Photron frames, naturally sorted", f"- Annotation source: {annotation_source}", "", "## Detection Results", "", "| Confidence | Detected | Missed | Detection Rate |", "|---:|---:|---:|---:|"]
    for threshold in summary.get("thresholds", DEFAULT_THRESHOLDS):
        item = summary.get("by_threshold", {}).get(str(threshold))
        if not item:
            lines.append(f"| {threshold} | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE |")
        else:
            lines.append(f"| {threshold} | {item['frames_with_detection']} | {item['frames_without_detection']} | {item['detection_rate']:.4f} |")
    for title, key in [("Missed Frames", "missed_frames"), ("Multiple Detection Frames", "multiple_detection_frames")]:
        lines.extend(["", f"## {title}", ""])
        entries = []
        for item in summary.get("by_threshold", {}).values():
            entries.extend(item.get(key, []))
        lines.append(", ".join(sorted(set(entries), key=lambda name: natural_frame_key(Path(name)))) if entries else "NOT AVAILABLE")
    lines.extend(["", "## Confidence Distribution", "", "Per-frame confidence statistics are stored in `artifacts/model_audit/summary.json`. They are reported only when the real frame sequence is available.", "", "## Localization", "", "NOT CALCULATED: the repository documents classical-CV pseudo-labels/reference labels, not independent ground truth. Precision, recall, F1, IoU, and localization error are intentionally not reported.", "", "## Limitations", "", "- The official dataset has no independent object-detection ground truth.", "- Existing labels were generated by background subtraction and contour detection and must be treated as reference labels/auto-labels.", "- The documented evaluation uses a single continuous shot and has data-leakage risk because training and validation use the same sequence.", "- No synthetic data or substitute video is used by this audit.", "", "## Conclusion", "", "This audit is diagnostic. It describes detector behavior at configured confidence thresholds and does not establish independent generalization or accuracy without trustworthy ground truth."])
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--frames-dir", type=Path, default=Path("data/yolo_dataset/images/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/model_audit"))
    parser.add_argument("--report", type=Path, default=Path("docs/model_audit.md"))
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        frames = discover_frames(args.frames_dir)
    except (FileNotFoundError, ValueError) as exc:
        summary = {"status": "NOT RUN", "reason": str(exc), "model_path": str(args.model), "frame_count": 0, "thresholds": args.thresholds, "by_threshold": {}}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_records_csv([], args.output_dir / "results.csv")
        write_summary_json(summary, args.output_dir / "summary.json")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_markdown(summary, False, str(args.frames_dir), "reference labels/auto-labels documented in docs/pseudo_labeling.md"), encoding="utf-8")
        print(f"AUDIT NOT RUN: {exc}")
        return 2
    records, summary = run_audit(args.model, frames, args.thresholds)
    summary["status"] = "COMPLETED"
    summary["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_records_csv(records, args.output_dir / "results.csv")
    write_summary_json(summary, args.output_dir / "summary.json")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_markdown(summary, True, str(args.frames_dir), "reference labels/auto-labels documented in docs/pseudo_labeling.md"), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
