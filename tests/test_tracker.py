"""Tests for the IOU/centroid face tracker."""

from src.detector import Detection
from src.tracker import FaceTracker


def _det(x1, y1, x2, y2):
    return Detection(box=(float(x1), float(y1), float(x2), float(y2)), confidence=0.9)


def test_new_detection_creates_track(config):
    tracker = FaceTracker(config)
    active, exited = tracker.update([_det(10, 10, 50, 50)])
    assert exited == []
    assert len(active) == 1
    assert active[0].track_id == 1


def test_same_region_keeps_same_track_id(config):
    tracker = FaceTracker(config)
    active1, _ = tracker.update([_det(10, 10, 50, 50)])
    active2, _ = tracker.update([_det(12, 11, 52, 51)])  # moved slightly
    assert active1[0].track_id == active2[0].track_id


def test_two_faces_get_distinct_track_ids(config):
    tracker = FaceTracker(config)
    active, _ = tracker.update([_det(10, 10, 50, 50), _det(300, 10, 340, 50)])
    ids = {t.track_id for t in active}
    assert len(ids) == 2


def test_track_exits_after_max_disappeared(config):
    tracker = FaceTracker(config)
    tracker.update([_det(10, 10, 50, 50)])
    max_disappeared = config.get("tracking", "max_disappeared_frames")
    exited = []
    for _ in range(max_disappeared + 2):
        active, exited = tracker.update([])  # no detections at all
        if exited:
            break  # stop as soon as the track is retired
    assert len(exited) == 1
    assert exited[0].track_id == 1
    assert active == []