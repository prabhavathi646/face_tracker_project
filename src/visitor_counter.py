"""Unique visitor counting.

A visitor is "unique" if their Face ID was registered exactly once. Since
``registration.py`` guarantees one Face ID per distinct person (new
registration only happens when no existing embedding matches above the
similarity threshold), the unique visitor count is simply the number of
distinct Face IDs ever registered -- tracked here both in-memory (fast,
authoritative during a run) and backed by the database (persisted,
authoritative across runs).
"""

from __future__ import annotations

import logging
from typing import Set

from src.database import Database

logger = logging.getLogger("face_tracker")


class VisitorCounter:
    """Tracks the running count of unique visitors seen so far."""

    def __init__(self, database: Database) -> None:
        self.db = database
        self._seen_face_ids: Set[str] = {fid for fid, _ in database.get_all_embeddings()}

    def register_new_visitor(self, face_id: str) -> None:
        """Mark a Face ID as a (new) unique visitor."""
        if face_id in self._seen_face_ids:
            logger.debug("Face %s already counted; skipping duplicate count.", face_id)
            return
        self._seen_face_ids.add(face_id)
        logger.info("Unique visitor count incremented -> %d (new: %s)",
                    len(self._seen_face_ids), face_id)

    def count(self) -> int:
        """Return the current in-memory unique visitor count."""
        return len(self._seen_face_ids)

    def persisted_count(self) -> int:
        """Return the authoritative count directly from the database."""
        return self.db.get_unique_visitor_count()
