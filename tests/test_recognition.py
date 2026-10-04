"""Tests for embedding matching logic (no model download required)."""

import numpy as np

from src.recognizer import find_best_match


def _unit(vec):
    vec = np.asarray(vec, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def test_exact_match_returns_face_id():
    emb = _unit([1.0, 0.0, 0.0])
    known = [("FACE-aaaa1111", _unit([1.0, 0.0, 0.0]))]
    face_id, score = find_best_match(emb, known, threshold=0.5)
    assert face_id == "FACE-aaaa1111"
    assert score > 0.99


def test_below_threshold_returns_none():
    emb = _unit([1.0, 0.0])
    known = [("FACE-aaaa1111", _unit([0.0, 1.0]))]  # orthogonal -> score 0
    face_id, score = find_best_match(emb, known, threshold=0.45)
    assert face_id is None
    assert score < 0.45


def test_empty_known_returns_none():
    face_id, _ = find_best_match(_unit([1.0, 2.0]), [], threshold=0.3)
    assert face_id is None


def test_best_of_multiple_is_picked():
    emb = _unit([1.0, 0.0, 0.0])
    known = [
        ("FACE-far00000", _unit([0.0, 1.0, 0.0])),
        ("FACE-near0000", _unit([0.9, 0.1, 0.0])),
        ("FACE-mid00000", _unit([0.5, 0.5, 0.0])),
    ]
    face_id, score = find_best_match(emb, known, threshold=0.5)
    assert face_id == "FACE-near0000"
    assert score > 0.8


def test_duplicate_registration_is_prevented_by_matching():
    """The same embedding seen twice must resolve to the same face id."""
    emb = _unit([0.3, 0.4, 0.5])
    known = [("FACE-dup00000", emb)]
    first_id, _ = find_best_match(emb, known, threshold=0.45)
    second_id, _ = find_best_match(emb, known, threshold=0.45)
    assert first_id == second_id == "FACE-dup00000"