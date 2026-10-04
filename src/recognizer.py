"""Face recognition using InsightFace / ArcFace embeddings.

Responsibilities:
    * Generate a 512-d ArcFace embedding for a cropped face image.
    * Compare a new embedding against all previously registered embeddings
      using cosine similarity to find the best match.
    * Apply a configurable similarity threshold to decide "known" vs "new".

We intentionally do NOT use the ``face_recognition`` library per project
requirements; InsightFace's ``FaceAnalysis`` app (ArcFace buffalo_l model)
is used instead and runs on CPU by default (ONNXRuntime CPUExecutionProvider).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from src.config import Config
from src.utils import cosine_similarity

logger = logging.getLogger("face_tracker")


class RecognizerLoadError(Exception):
    """Raised when the InsightFace model fails to load."""


def find_best_match(embedding: np.ndarray,
                    known_embeddings: List[Tuple[str, np.ndarray]],
                    threshold: float) -> Tuple[Optional[str], float]:
    """Pure matching logic: best cosine similarity above ``threshold``.

    Kept as a module-level function so it can be unit-tested without
    loading the InsightFace model. Returns ``(face_id, similarity)``;
    ``face_id`` is ``None`` when nothing clears the threshold.
    """
    best_face_id: Optional[str] = None
    best_score = -1.0
    for face_id, known_vec in known_embeddings:
        score = cosine_similarity(embedding, known_vec)
        if score > best_score:
            best_score = score
            best_face_id = face_id

    if best_face_id is not None and best_score >= threshold:
        return best_face_id, best_score
    return None, max(best_score, 0.0)


class FaceRecognizer:
    """Wraps InsightFace's FaceAnalysis app for embedding + matching."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model_name = config.get("recognition", "model_name", default="buffalo_l")
        self.threshold = config.get("recognition", "similarity_threshold", default=0.45)
        self.providers = config.get(
            "recognition", "providers", default=["CPUExecutionProvider"]
        )
        det_size = config.get("recognition", "det_size", default=[640, 640])
        self.det_size = tuple(det_size)
        self._app = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RecognizerLoadError(
                "insightface package is not installed. Run `pip install insightface`."
            ) from exc
        try:
            self._app = FaceAnalysis(name=self.model_name, providers=self.providers)
            self._app.prepare(ctx_id=0, det_size=self.det_size)
            logger.info("InsightFace recognizer loaded (model=%s)", self.model_name)
        except Exception as exc:  # noqa: BLE001
            raise RecognizerLoadError(
                f"Failed to load InsightFace model '{self.model_name}': {exc}"
            ) from exc

    def get_embedding(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """Run InsightFace on a cropped face image, return its embedding.

        Returns ``None`` if no face could be analyzed in the crop (e.g.
        the crop is too small, blurry, or badly aligned).
        """
        if self._app is None:
            logger.error("Recognizer called before model was loaded.")
            return None
        try:
            faces = self._app.get(face_crop)
        except Exception as exc:  # noqa: BLE001
            logger.exception("InsightFace embedding extraction failed: %s", exc)
            return None
        if not faces:
            logger.debug("InsightFace found no embeddable face in crop.")
            return None
        # If multiple faces are found in the crop (rare, since YOLO already
        # isolated one face), pick the one with highest detection score.
        best = max(faces, key=lambda f: getattr(f, "det_score", 0.0))
        return np.asarray(best.normed_embedding, dtype=np.float32)

    def match(self, embedding: np.ndarray,
              known_embeddings: List[Tuple[str, np.ndarray]]) -> Tuple[Optional[str], float]:
        """Find the best matching known face for a new embedding.

        Returns ``(face_id, similarity)``. ``face_id`` is ``None`` when the
        best similarity found is below the configured threshold, meaning
        this is treated as a new/unknown face.
        """
        face_id, score = find_best_match(embedding, known_embeddings, self.threshold)
        if face_id is not None:
            logger.debug("Matched embedding to %s (similarity=%.3f)", face_id, score)
        else:
            logger.debug("No match above threshold (best=%.3f < %.3f)", score, self.threshold)
        return face_id, score
