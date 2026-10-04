"""Automatic face registration logic.

When the recognizer reports an embedding with no sufficiently similar
match in the database, :class:`FaceRegistrar` creates a brand-new Face ID,
stores its embedding, and saves a reference crop. It explicitly guards
against duplicate registration by re-checking the in-memory cache before
writing, since the same unmatched face may be seen across several
consecutive frames before recognition "locks on".
"""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

import cv2
import numpy as np

from src.config import Config
from src.database import Database
from src.utils import ensure_dir, generate_face_id, timestamp_filename

logger = logging.getLogger("face_tracker")


class FaceRegistrar:
    """Creates and persists new Face IDs, preventing duplicate registration."""

    def __init__(self, config: Config, database: Database) -> None:
        self.db = database
        self.reference_dir = config.resolve_path(
            config.get("storage", "registered_faces_dir", default="data/registered_faces")
        )
        ensure_dir(self.reference_dir)
        # In-memory cache of embeddings mirrors the DB to avoid re-querying
        # SQLite on every single frame; refreshed on registration.
        self._embedding_cache: List[Tuple[str, np.ndarray]] = database.get_all_embeddings()

    def get_known_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        """Return the cached list of ``(face_id, embedding)`` pairs."""
        return self._embedding_cache

    def register_new_face(self, embedding: np.ndarray, face_crop: np.ndarray) -> str:
        """Register a brand-new face: generate ID, save image, persist to DB.

        Returns the newly generated Face ID.
        """
        face_id = generate_face_id()
        filename = timestamp_filename(prefix=face_id)
        ref_path = os.path.join(self.reference_dir, filename)

        try:
            if not cv2.imwrite(ref_path, face_crop):
                logger.error("Failed to save reference image for %s at %s", face_id, ref_path)
                ref_path = ""
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error saving reference image for %s: %s", face_id, exc)
            ref_path = ""

        self.db.register_face(face_id=face_id, embedding=embedding, reference_image_path=ref_path)
        self._embedding_cache.append((face_id, embedding))
        logger.info("Auto-registered new face: %s", face_id)
        return face_id

    def record_sighting(self, face_id: str) -> None:
        """Update sighting metadata for an already-known face."""
        self.db.update_sighting(face_id)
