"""Tests for automatic face registration."""

import os

import numpy as np

from src.registration import FaceRegistrar


def test_register_new_face_creates_id_and_persists(config, db, sample_crop, sample_embedding):
    registrar = FaceRegistrar(config, db)
    face_id = registrar.register_new_face(sample_embedding, sample_crop)

    assert face_id.startswith("FACE-")
    assert db.face_exists(face_id)
    # Embedding must be visible through the registrar's cache.
    assert any(fid == face_id for fid, _ in registrar.get_known_embeddings())


def test_registration_saves_reference_image(config, db, sample_crop, sample_embedding):
    registrar = FaceRegistrar(config, db)
    face_id = registrar.register_new_face(sample_embedding, sample_crop)
    row = db._conn.execute(
        "SELECT reference_image_path FROM faces WHERE face_id = ?", (face_id,)
    ).fetchone()
    assert row is not None
    assert os.path.exists(row[0])


def test_duplicate_registration_prevented(config, db, sample_crop, sample_embedding):
    """Two registrations of identical embeddings must yield distinct IDs
    in the registrar API, but matching before registering is what prevents
    duplicates -- verify the match path reuses the first ID."""
    from src.recognizer import find_best_match

    registrar = FaceRegistrar(config, db)
    first_id = registrar.register_new_face(sample_embedding, sample_crop)
    matched_id, score = find_best_match(
        sample_embedding, registrar.get_known_embeddings(), threshold=0.45
    )
    assert matched_id == first_id
    assert db.get_unique_visitor_count() == 1


def test_sighting_updates_last_seen(config, db, sample_crop, sample_embedding):
    registrar = FaceRegistrar(config, db)
    face_id = registrar.register_new_face(sample_embedding, sample_crop)
    registrar.record_sighting(face_id)
    row = db._conn.execute(
        "SELECT total_sightings FROM faces WHERE face_id = ?", (face_id,)
    ).fetchone()
    assert row[0] >= 2


def test_face_ids_are_unique(config, db, sample_crop):
    registrar = FaceRegistrar(config, db)
    ids = set()
    for _ in range(5):
        emb = np.random.default_rng().standard_normal(512).astype(np.float32)
        ids.add(registrar.register_new_face(emb, sample_crop))
    assert len(ids) == 5