"""Logging setup and entry/exit event persistence.

This module owns two related responsibilities that are intentionally kept
together because every ENTRY/EXIT event must *both* be written to
``events.log`` *and* persisted to SQLite with its cropped image:

1. :func:`setup_logging` configures Python's ``logging`` module to write to
   the mandatory ``events.log`` file as well as the console.
2. :class:`EventLogger` saves cropped face images to the dated
   ``logs/entries/YYYY-MM-DD`` or ``logs/exits/YYYY-MM-DD`` folders and
   records the event row in the database.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from src.config import Config
from src.database import Database
from src.utils import dated_subdir, timestamp_filename

logger = logging.getLogger("face_tracker")


def setup_logging(config: Config) -> logging.Logger:
    """Configure the root ``face_tracker`` logger.

    Writes to the mandatory ``events.log`` file (resolved from config, at
    project root by default) and mirrors messages to the console.
    """
    log_file = config.resolve_path(config.get("logging", "log_file", default="events.log"))
    level_name = config.get("logging", "log_level", default="INFO")
    console_level_name = config.get("logging", "console_level", default="INFO")

    log = logging.getLogger("face_tracker")
    log.setLevel(getattr(logging, level_name.upper(), logging.INFO))
    log.handlers.clear()  # avoid duplicate handlers on re-init (e.g. tests)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(log.level)
    log.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    console_handler.setLevel(getattr(logging, console_level_name.upper(), logging.INFO))
    log.addHandler(console_handler)

    log.propagate = False
    log.info("Logging initialized -> %s", log_file)
    return log


class EventLogger:
    """Persists ENTRY/EXIT events: saves the cropped image and DB row."""

    def __init__(self, config: Config, database: Database) -> None:
        self.config = config
        self.db = database
        self.entries_dir = config.resolve_path(
            config.get("storage", "entries_dir", default="logs/entries")
        )
        self.exits_dir = config.resolve_path(
            config.get("storage", "exits_dir", default="logs/exits")
        )

    def _save_crop(self, base_dir: str, face_crop: np.ndarray, face_id: str) -> Optional[str]:
        """Save a face crop into ``base_dir/YYYY-MM-DD/`` and return its path."""
        try:
            target_dir = dated_subdir(base_dir)
            filename = timestamp_filename(prefix=face_id)
            full_path = os.path.join(target_dir, filename)
            success = cv2.imwrite(full_path, face_crop)
            if not success:
                logger.error("cv2.imwrite failed for %s", full_path)
                return None
            return full_path
        except Exception as exc:  # noqa: BLE001 - must not crash the pipeline
            logger.exception("Failed to save face crop image: %s", exc)
            return None

    def log_entry(self, face_id: str, track_id: int, face_crop: np.ndarray,
                   similarity: float = 0.0, is_new_registration: bool = False) -> Optional[int]:
        """Record one ENTRY event: save image + insert DB row + log line."""
        image_path = self._save_crop(self.entries_dir, face_crop, face_id)
        timestamp = datetime.now().isoformat()
        try:
            event_id = self.db.insert_event(
                face_id=face_id,
                track_id=track_id,
                event_type="ENTRY",
                timestamp=timestamp,
                image_path=image_path or "",
                similarity=similarity,
            )
            logger.info(
                "ENTRY event | face_id=%s track_id=%s new_registration=%s image=%s",
                face_id, track_id, is_new_registration, image_path,
            )
            return event_id
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to record ENTRY event in DB: %s", exc)
            return None

    def log_exit(self, face_id: str, track_id: int, face_crop: Optional[np.ndarray],
                  similarity: float = 0.0) -> Optional[int]:
        """Record one EXIT event: save image (if available) + insert DB row."""
        image_path = None
        if face_crop is not None:
            image_path = self._save_crop(self.exits_dir, face_crop, face_id)
        timestamp = datetime.now().isoformat()
        try:
            event_id = self.db.insert_event(
                face_id=face_id,
                track_id=track_id,
                event_type="EXIT",
                timestamp=timestamp,
                image_path=image_path or "",
                similarity=similarity,
            )
            logger.info(
                "EXIT event | face_id=%s track_id=%s image=%s",
                face_id, track_id, image_path,
            )
            return event_id
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to record EXIT event in DB: %s", exc)
            return None
