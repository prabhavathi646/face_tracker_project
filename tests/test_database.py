"""Tests for SQLite database operations."""

import numpy as np


def test_register_face_inserts_rows(db, sample_embedding):
    db.register_face("FACE-abc12345", sample_embedding, "ref.jpg")
    assert db.face_exists("FACE-abc12345")
    embeddings = db.get_all_embeddings()
    assert len(embeddings) == 1
    face_id, vec = embeddings[0]
    assert face_id == "FACE-abc12345"
    np.testing.assert_allclose(vec, sample_embedding, rtol=1e-6)


def test_unique_visitor_count(db, sample_embedding):
    assert db.get_unique_visitor_count() == 0
    db.register_face("FACE-one1111", sample_embedding, "a.jpg")
    db.register_face("FACE-two2222", sample_embedding, "b.jpg")
    assert db.get_unique_visitor_count() == 2


def test_update_sighting_increments(db, sample_embedding):
    db.register_face("FACE-abc12345", sample_embedding, "ref.jpg")
    db.update_sighting("FACE-abc12345")
    import sqlite3
    cur = db._conn.execute("SELECT total_sightings FROM faces WHERE face_id = ?",
                           ("FACE-abc12345",))
    assert cur.fetchone()[0] == 2


def test_insert_and_count_events(db):
    db.register_face("FACE-abc12345", np.ones(8, dtype=np.float32), "ref.jpg")
    db.insert_event("FACE-abc12345", 1, "ENTRY", "2026-01-01T00:00:00", "e.jpg", 0.9)
    db.insert_event("FACE-abc12345", 1, "EXIT", "2026-01-01T00:01:00", "x.jpg", 0.9)
    entries, exits = db.get_event_counts()
    assert entries == 1
    assert exits == 1


def test_parameterized_queries_protect_against_injection(db, sample_embedding):
    weird_id = "FACE-'; DROP TABLE faces;--"
    db.register_face(weird_id, sample_embedding, "ref.jpg")
    # Tables must still exist after inserting a hostile-looking ID.
    assert db.face_exists(weird_id)
    assert db.get_unique_visitor_count() == 1