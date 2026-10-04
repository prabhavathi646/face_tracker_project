"""Tests for ENTRY/EXIT event creation (image + database rows)."""

import os

from src.event_logger import EventLogger


def test_entry_event_creates_db_row_and_image(config, db, sample_crop):
    db.register_face("FACE-abc12345", sample_crop.mean(axis=(0, 1)).astype("float32"), "r.jpg")
    logger_ = EventLogger(config, db)
    event_id = logger_.log_entry("FACE-abc12345", track_id=1, face_crop=sample_crop)
    assert event_id is not None

    row = db._conn.execute(
        "SELECT event_type, image_path FROM events WHERE id = ?", (event_id,)
    ).fetchone()
    assert row[0] == "ENTRY"
    assert row[1] and os.path.exists(row[1])
    # Path must follow logs/entries/YYYY-MM-DD/ layout.
    assert f"entries{os.sep}" in row[1]


def test_exit_event_creates_db_row_and_image(config, db, sample_crop):
    db.register_face("FACE-abc12345", sample_crop.mean(axis=(0, 1)).astype("float32"), "r.jpg")
    logger_ = EventLogger(config, db)
    event_id = logger_.log_exit("FACE-abc12345", track_id=1, face_crop=sample_crop)
    assert event_id is not None
    row = db._conn.execute(
        "SELECT event_type, image_path FROM events WHERE id = ?", (event_id,)
    ).fetchone()
    assert row[0] == "EXIT"
    assert row[1] and os.path.exists(row[1])
    assert f"exits{os.sep}" in row[1]


def test_exit_without_crop_still_logs(config, db):
    db.register_face("FACE-abc12345", __import__("numpy").ones(8, dtype="float32"), "r.jpg")
    logger_ = EventLogger(config, db)
    event_id = logger_.log_exit("FACE-abc12345", track_id=2, face_crop=None)
    assert event_id is not None
    row = db._conn.execute(
        "SELECT image_path FROM events WHERE id = ?", (event_id,)
    ).fetchone()
    assert row[0] == ""  # empty path, but the event is still recorded


def test_exactly_one_entry_per_face_track(config, db, sample_crop):
    """The main loop guards entry_logged; here we verify counts stay correct
    when only one entry is logged per track."""
    db.register_face("FACE-abc12345", sample_crop.mean(axis=(0, 1)).astype("float32"), "r.jpg")
    logger_ = EventLogger(config, db)
    logger_.log_entry("FACE-abc12345", track_id=1, face_crop=sample_crop)
    entries, _ = db.get_event_counts()
    assert entries == 1