"""Shared utility helpers used across the face tracker modules.

Keeping these pure, dependency-light helper functions in one place avoids
code duplication between the detector, tracker and recognizer modules.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Iterable, Tuple

import cv2
import numpy as np


def generate_face_id() -> str:
    """Generate a short, unique, human-readable Face ID.

    Format: ``FACE-<8 hex chars>`` e.g. ``FACE-3b9ac9ff``.
    UUID4 guarantees practical uniqueness without a central counter.
    """
    return f"FACE-{uuid.uuid4().hex[:8]}"


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D embedding vectors.

    Returns a value in ``[-1, 1]``. Embeddings from InsightFace are
    already L2-normalized, but we normalize again defensively.
    """
    a = vec_a.astype(np.float32).flatten()
    b = vec_b.astype(np.float32).flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def iou(box_a: Tuple[float, float, float, float],
        box_b: Tuple[float, float, float, float]) -> float:
    """Compute Intersection-over-Union of two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def box_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Return the ``(cx, cy)`` center point of a bounding box."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two 2-D points."""
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def ensure_dir(path: str) -> str:
    """Create a directory (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)
    return path


def dated_subdir(base_dir: str, when: datetime | None = None) -> str:
    """Return (and create) ``base_dir/YYYY-MM-DD`` for the given timestamp."""
    when = when or datetime.now()
    sub = os.path.join(base_dir, when.strftime("%Y-%m-%d"))
    return ensure_dir(sub)


def timestamp_filename(prefix: str, ext: str = "jpg") -> str:
    """Generate a collision-resistant, sortable filename."""
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{stamp}.{ext}"


def clamp_box(box: Iterable[float], width: int, height: int) -> Tuple[int, int, int, int]:
    """Clamp a float bounding box into valid integer pixel coordinates."""
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(0, min(int(round(x2)), width))
    y2 = max(0, min(int(round(y2)), height))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def prepare_face_crop(frame: np.ndarray, box: Tuple[float, float, float, float],
                       margin: float = 0.4, min_size: int = 160) -> np.ndarray:
    """Crop a face with breathing room and guarantee a usable input size.

    InsightFace's own detector expects some context around the face — a
    tight YOLO box (face only) often yields *no* detected face. This pads
    the box by ``margin`` (fraction of box size) on each side, clamps it
    to the frame, then upscales small crops to at least ``min_size`` px
    on the short side (INTER_CUBIC) so tiny distant faces still embed.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw = (x2 - x1) * margin
    bh = (y2 - y1) * margin
    padded = (x1 - bw, y1 - bh, x2 + bw, y2 + bh)
    px1, py1, px2, py2 = clamp_box(padded, width, height)
    crop = frame[py1:py2, px1:px2]

    if crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    short_side = min(h, w)
    if short_side < min_size:
        scale = min_size / short_side
        crop = cv2.resize(crop, (max(1, int(round(w * scale))),
                                  max(1, int(round(h * scale)))),
                           interpolation=cv2.INTER_CUBIC)
    return crop
