from pathlib import Path

from model_audit import Detection, discover_frames, make_frame_record, natural_frame_key, parse_detections, summarize


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class FakeBoxes:
    xyxy = FakeTensor([[10, 20, 30, 40], [0, 0, 5, 5]])
    conf = FakeTensor([0.25, 0.90])


class FakeResult:
    boxes = FakeBoxes()


def test_natural_frame_order():
    paths = [Path("frame_10.png"), Path("frame_2.png"), Path("frame_1.png")]
    assert [path.name for path in sorted(paths, key=natural_frame_key)] == ["frame_1.png", "frame_2.png", "frame_10.png"]


def test_parse_all_detections_and_select_highest_confidence():
    detections = parse_detections(FakeResult())
    assert len(detections) == 2
    assert detections[0].center_x == 20
    assert detections[0].width == 20
    record = make_frame_record(0.25, 0, "frame_000.png", detections)
    assert record.selected_detection_index == 1
    assert record.confidence == 0.90
    assert record.detection_count == 2
    assert record.status == "DETECTED"
    assert len(record.detections) == 2


def test_statistics_and_miss_detection():
    detected = make_frame_record(0.25, 0, "frame_000.png", [Detection(0, 0.8, 0, 0, 10, 10, 5, 5, 10, 10)])
    multiple = make_frame_record(0.25, 1, "frame_001.png", [Detection(0, 0.4, 0, 0, 10, 10, 5, 5, 10, 10), Detection(1, 0.6, 1, 1, 11, 11, 6, 6, 10, 10)])
    missed = make_frame_record(0.25, 2, "frame_002.png", [])
    stats = summarize([detected, multiple, missed])
    assert stats["total_frames"] == 3
    assert stats["frames_with_detection"] == 2
    assert stats["frames_without_detection"] == 1
    assert stats["detection_rate"] == 2 / 3
    assert stats["missed_frames"] == ["frame_002.png"]
    assert stats["multiple_detection_frames"] == ["frame_001.png"]
    assert stats["confidence_min"] == 0.6
    assert stats["confidence_max"] == 0.8


def test_discover_frames_recursively(tmp_path):
    (tmp_path / "nested").mkdir()
    for name in ["frame_10.png", "frame_2.png", "frame_1.png", "ignore.txt"]:
        (tmp_path / "nested" / name).write_bytes(b"data")
    assert [path.name for path in discover_frames(tmp_path)] == ["frame_1.png", "frame_2.png", "frame_10.png"]
