"""Tests for the unique visitor counter."""

import numpy as np

from src.visitor_counter import VisitorCounter


def test_starts_with_zero(db):
    assert VisitorCounter(db).count() == 0


def test_increments_once_per_unique_face(db):
    counter = VisitorCounter(db)
    emb = np.ones(8, dtype=np.float32)
    db.register_face("FACE-one1111", emb, "a.jpg")
    counter.register_new_visitor("FACE-one1111")
    counter.register_new_visitor("FACE-one1111")  # duplicate: must not count
    assert counter.count() == 1


def test_multiple_visitors(db):
    counter = VisitorCounter(db)
    emb = np.ones(8, dtype=np.float32)
    for fid in ("FACE-one1111", "FACE-two2222", "FACE-three33"):
        db.register_face(fid, emb, f"{fid}.jpg")
        counter.register_new_visitor(fid)
    assert counter.count() == 3


def test_persisted_count_matches_database(db, sample_embedding):
    db.register_face("FACE-one1111", sample_embedding, "a.jpg")
    db.register_face("FACE-two2222", sample_embedding, "b.jpg")
    counter = VisitorCounter(db)
    # Counter seeds itself from existing DB embeddings.
    assert counter.count() == 2
    assert counter.persisted_count() == 2