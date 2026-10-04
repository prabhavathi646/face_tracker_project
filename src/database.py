"""SQLite persistence layer for the Intelligent Face Tracker.

Three tables are used:

``faces``
    One row per unique registered person (Face ID, registration timestamp,
    reference image path, total sightings, last seen time).

``embeddings``
    One or more embedding vectors per face (stored as serialized float32
    bytes). Keeping embeddings in a separate table allows a future
    improvement where a person has multiple embeddings for more robust
    matching (e.g. captured at different angles/lighting).

``events``
    One row per ENTRY/EXIT event with face id, tracking id, type,
    timestamp, image path and the similarity score used for the match.

All access goes through parameterized queries to avoid SQL injection and
to keep the code consistent with the rest of the project's conventions.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Generator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("face_tracker")

SCHEMA = """
CREATE TABLE IF NOT EXISTS faces (
    face_id TEXT PRIMARY KEY,
    registered_at TEXT NOT NULL,
    reference_image_path TEXT,
    total_sightings INTEGER NOT NULL DEFAULT 1,
    last_seen_at TEXT
);

CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (face_id) REFERENCES faces (face_id)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id TEXT NOT NULL,
    track_id INTEGER,
    event_type TEXT NOT NULL CHECK (event_type IN ('ENTRY', 'EXIT')),
    timestamp TEXT NOT NULL,
    image_path TEXT,
    similarity REAL,
    FOREIGN KEY (face_id) REFERENCES faces (face_id)
);

CREATE INDEX IF NOT EXISTS idx_events_face_id ON events (face_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_face_id ON embeddings (face_id);
"""


class Database:
    """Thin SQLite wrapper providing the operations the pipeline needs."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._init_schema()

    def _init_schema(self) -> None:
        try:
            self._conn.executescript(SCHEMA)
            self._conn.commit()
            logger.info("Database schema ready at %s", self.db_path)
        except sqlite3.Error as exc:
            logger.exception("Failed to initialize database schema: %s", exc)
            raise

    @contextmanager
    def _cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except sqlite3.Error:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    # ------------------------------------------------------------------ #
    # Face registration
    # ------------------------------------------------------------------ #
    def register_face(self, face_id: str, embedding: np.ndarray,
                       reference_image_path: str,
                       registered_at: Optional[str] = None) -> None:
        """Insert a brand-new face and its first embedding."""
        registered_at = registered_at or datetime.now().isoformat()
        try:
            with self._cursor() as cur:
                cur.execute(
                    """INSERT INTO faces (face_id, registered_at, reference_image_path,
                                           total_sightings, last_seen_at)
                       VALUES (?, ?, ?, 1, ?)""",
                    (face_id, registered_at, reference_image_path, registered_at),
                )
                cur.execute(
                    """INSERT INTO embeddings (face_id, embedding, created_at)
                       VALUES (?, ?, ?)""",
                    (face_id, embedding.astype(np.float32).tobytes(), registered_at),
                )
            logger.info("Registered new face %s (image=%s)", face_id, reference_image_path)
        except sqlite3.Error as exc:
            logger.exception("Database error while registering face %s: %s", face_id, exc)
            raise

    def update_sighting(self, face_id: str, seen_at: Optional[str] = None) -> None:
        """Increment sighting count and refresh last_seen_at for a known face."""
        seen_at = seen_at or datetime.now().isoformat()
        try:
            with self._cursor() as cur:
                cur.execute(
                    """UPDATE faces SET total_sightings = total_sightings + 1,
                                         last_seen_at = ?
                       WHERE face_id = ?""",
                    (seen_at, face_id),
                )
        except sqlite3.Error as exc:
            logger.exception("Database error updating sighting for %s: %s", face_id, exc)
            raise

    def get_all_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        """Return ``[(face_id, embedding_vector), ...]`` for all known faces."""
        try:
            with self._cursor() as cur:
                cur.execute("SELECT face_id, embedding FROM embeddings")
                rows = cur.fetchall()
            return [(face_id, np.frombuffer(blob, dtype=np.float32)) for face_id, blob in rows]
        except sqlite3.Error as exc:
            logger.exception("Database error fetching embeddings: %s", exc)
            return []

    def face_exists(self, face_id: str) -> bool:
        with self._cursor() as cur:
            cur.execute("SELECT 1 FROM faces WHERE face_id = ?", (face_id,))
            return cur.fetchone() is not None

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #
    def insert_event(self, face_id: str, track_id: Optional[int], event_type: str,
                      timestamp: str, image_path: str, similarity: float) -> int:
        """Insert one ENTRY/EXIT event row and return its new row id."""
        try:
            with self._cursor() as cur:
                cur.execute(
                    """INSERT INTO events (face_id, track_id, event_type, timestamp,
                                            image_path, similarity)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (face_id, track_id, event_type, timestamp, image_path, similarity),
                )
                return cur.lastrowid
        except sqlite3.Error as exc:
            logger.exception("Database error inserting %s event for %s: %s",
                              event_type, face_id, exc)
            raise

    # ------------------------------------------------------------------ #
    # Reporting / counting
    # ------------------------------------------------------------------ #
    def get_unique_visitor_count(self) -> int:
        """The number of unique visitors is simply the number of registered faces."""
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM faces")
            return int(cur.fetchone()[0])

    def get_event_counts(self) -> Tuple[int, int]:
        """Return ``(entry_count, exit_count)`` across all recorded events."""
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM events WHERE event_type = 'ENTRY'")
            entries = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM events WHERE event_type = 'EXIT'")
            exits = int(cur.fetchone()[0])
        return entries, exits

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        try:
            self._conn.close()
        except sqlite3.Error as exc:
            logger.warning("Error closing database connection: %s", exc)

