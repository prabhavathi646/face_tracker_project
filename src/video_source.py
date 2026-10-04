"""Video input abstraction: local file today, RTSP camera tomorrow.

The rest of the pipeline only ever calls :meth:`VideoSource.read` and never
needs to know whether frames come from a file on disk or a live RTSP
stream. Switching source is purely a ``config.json`` change
(``video_source.type``: ``"file"`` or ``"rtsp"``).
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config import Config

logger = logging.getLogger("face_tracker")


class VideoSourceError(Exception):
    """Raised when the configured video source cannot be opened/read."""


class VideoSource:
    """Wraps ``cv2.VideoCapture`` with reconnect logic for RTSP streams."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.source_type = config.get("video_source", "type", default="file")
        self.file_path = config.resolve_path(
            config.get("video_source", "file_path", default="data/sample_video.mp4")
        )
        self.rtsp_url = config.get("video_source", "rtsp_url", default="")
        self.reconnect_attempts = config.get(
            "video_source", "reconnect_attempts", default=5
        )
        self.reconnect_delay = config.get(
            "video_source", "reconnect_delay_seconds", default=2
        )
        self._cap: Optional[cv2.VideoCapture] = None
        self._open()

    def _target(self) -> str:
        return self.rtsp_url if self.source_type == "rtsp" else self.file_path

    def _open(self) -> None:
        target = self._target()
        if self.source_type == "file":
            import os
            if not os.path.exists(target):
                raise VideoSourceError(f"Video file not found: {target}")
        logger.info("Opening video source (%s): %s", self.source_type, target)
        cap = cv2.VideoCapture(target)
        if not cap.isOpened():
            raise VideoSourceError(f"Unable to open video source: {target}")
        self._cap = cap

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read one frame. Attempts reconnect on failure for RTSP sources."""
        if self._cap is None:
            return False, None
        ok, frame = self._cap.read()
        if ok:
            return True, frame

        if self.source_type != "rtsp":
            return False, None  # end of local file, normal termination

        logger.warning("Lost RTSP frame; attempting reconnect...")
        for attempt in range(1, self.reconnect_attempts + 1):
            time.sleep(self.reconnect_delay)
            try:
                self._cap.release()
                self._open()
                ok, frame = self._cap.read()
                if ok:
                    logger.info("RTSP reconnect succeeded on attempt %d", attempt)
                    return True, frame
            except VideoSourceError as exc:
                logger.warning("RTSP reconnect attempt %d failed: %s", attempt, exc)
        logger.error("Exhausted RTSP reconnect attempts; giving up.")
        return False, None

    def get_fps(self) -> float:
        if self._cap is None:
            return 0.0
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 25.0

    def get_frame_size(self) -> Tuple[int, int]:
        """Return ``(width, height)`` of the video frames."""
        if self._cap is None:
            return (0, 0)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            logger.info("Video source released.")
