"""YOLO-based face detection.

Uses Ultralytics YOLO (v8) with a face-detection-trained weights file
(configurable via ``config.json -> detection.model_path``, e.g.
``yolov8n-face.pt``). If a face-specific model is unavailable, a generic
YOLOv8 "person" model can still be used as a fallback for demos, but a
face-specific model is strongly recommended for accuracy.

Detection runs are frame-skipped according to ``detection.frame_skip`` to
keep the pipeline real-time: when a frame is skipped, the tracker simply
propagates the previous detections/tracks forward instead of recomputing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from src.config import Config

logger = logging.getLogger("face_tracker")


class DetectorLoadError(Exception):
    """Raised when the YOLO model fails to load."""


@dataclass
class Detection:
    """One detected face bounding box with confidence."""
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float


class FaceDetector:
    """Thin wrapper around an Ultralytics YOLO model for face detection."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model_path = config.resolve_path(
            config.get("detection", "model_path", default="models/yolov8n-face.pt")
        )
        self.conf_threshold = config.get("detection", "confidence_threshold", default=0.45)
        self.device = config.get("detection", "device", default="cpu")
        self.input_size = config.get("detection", "input_size", default=640)
        self.frame_skip = config.get("detection", "frame_skip", default=2)
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO  # imported lazily to speed up tests
        except ImportError as exc:
            raise DetectorLoadError(
                "ultralytics package is not installed. Run `pip install ultralytics`."
            ) from exc
        try:
            self._model = YOLO(self.model_path)
            logger.info("YOLO face detector loaded from %s (device=%s)",
                        self.model_path, self.device)
        except Exception as exc:  # noqa: BLE001
            raise DetectorLoadError(
                f"Failed to load YOLO model from {self.model_path}: {exc}"
            ) from exc

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLO inference on a single BGR frame and return detections."""
        if self._model is None:
            logger.error("Detector called before model was loaded.")
            return []
        try:
            results = self._model.predict(
                source=frame,
                imgsz=self.input_size,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("YOLO inference failed: %s", exc)
            return []

        detections: List[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box, conf in zip(boxes.xyxy.tolist(), boxes.conf.tolist()):
                x1, y1, x2, y2 = box
                detections.append(Detection(box=(x1, y1, x2, y2), confidence=float(conf)))
        logger.debug("Detected %d face(s) in frame", len(detections))
        return detections
