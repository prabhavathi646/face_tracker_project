"""Tests for shared utility helpers."""

import numpy as np

from src.utils import (box_center, clamp_box, cosine_similarity, euclidean_distance,
                        generate_face_id, iou, prepare_face_crop)


def test_iou_identical_boxes_is_one():
    box = (10.0, 10.0, 50.0, 50.0)
    assert abs(iou(box, box) - 1.0) < 1e-6


def test_iou_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_cosine_similarity_identical():
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_similarity(v, v) > 0.999


def test_cosine_similarity_zero_vector():
    assert cosine_similarity(np.zeros(3), np.ones(3)) == 0.0


def test_clamp_box_stays_in_bounds():
    x1, y1, x2, y2 = clamp_box((-50, -50, 500, 500), 100, 80)
    assert 0 <= x1 < x2 <= 100
    assert 0 <= y1 < y2 <= 80


def test_generate_face_id_format():
    fid = generate_face_id()
    assert fid.startswith("FACE-")
    assert len(fid) == len("FACE-") + 8


def test_box_center_and_distance():
    assert box_center((0, 0, 10, 10)) == (5.0, 5.0)
    assert euclidean_distance((0, 0), (3, 4)) == 5.0


def test_prepare_face_crop_pads_and_upscales():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Tiny tight box: padded result must be larger and short side >= min_size.
    crop = prepare_face_crop(frame, (100, 100, 130, 140), margin=0.2, min_size=160)
    h, w = crop.shape[:2]
    assert min(h, w) >= 160


def test_prepare_face_crop_clamps_at_frame_edge():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = prepare_face_crop(frame, (0, 0, 20, 20), margin=0.5, min_size=10)
    assert crop.size > 0
    assert crop.shape[0] <= 100 and crop.shape[1] <= 100