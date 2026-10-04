"""Lightweight multi-object tracker for face bounding boxes.

Implements a ByteTrack-style IOU + centroid-distance association without
extra heavyweight dependencies, keeping the hackathon project easy to
install and explain. Each track carries:

    * ``track_id``      - stable integer identity for the track's lifetime
    * ``face_id``        - persistent identity once recognized/registered
    * ``box``             - last known bounding box
    * ``disappeared``   - consecutive frames without a matching detection

A track that has not been matched for more than
``tracking.max_disappeared_frames`` is considered to have EXITED the frame
and is removed (triggering an EXIT event upstream in ``main.py``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import Config
from src.detector import Detection
from src.utils import box_center, euclidean_distance, iou

logger = logging.getLogger("face_tracker")


@dataclass
class Track:
    """Represents one actively tracked face across frames."""
    track_id: int
    box: Tuple[float, float, float, float]
    disappeared: int = 0
    face_id: Optional[str] = None
    entry_logged: bool = False
    best_similarity: float = 0.0
    frames_seen: int = 1
    last_crop: Optional[np.ndarray] = None  # most recent face crop, for EXIT images


class FaceTracker:
    """Associates detections across frames into persistent tracks."""

    def __init__(self, config: Config) -> None:
        self.max_disappeared = config.get(
            "tracking", "max_disappeared_frames", default=30
        )
        self.iou_threshold = config.get("tracking", "iou_match_threshold", default=0.3)
        self.max_distance = config.get("tracking", "max_distance_px", default=120)
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1

    def _new_track_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def update(self, detections: List[Detection]) -> Tuple[List[Track], List[Track]]:
        """Update tracks with the latest detections.

        Returns ``(active_tracks, exited_tracks)`` where ``exited_tracks``
        is the list of tracks removed this call because they disappeared
        for too long (candidates for EXIT events).
        """
        unmatched_detections = list(range(len(detections)))
        matched_track_ids: set = set()

        # Greedy matching: for each existing track, find the best detection
        # by IOU first, falling back to centroid distance for small/fast
        # motions where boxes may not overlap.
        for track_id, track in self.tracks.items():
            best_idx = -1
            best_iou = 0.0
            for idx in unmatched_detections:
                score = iou(track.box, detections[idx].box)
                if score > best_iou:
                    best_iou = score
                    best_idx = idx

            if best_idx != -1 and best_iou >= self.iou_threshold:
                track.box = detections[best_idx].box
                track.disappeared = 0
                track.frames_seen += 1
                unmatched_detections.remove(best_idx)
                matched_track_ids.add(track_id)
                continue

            # Fallback: nearest centroid within max_distance
            best_idx = -1
            best_dist = float("inf")
            for idx in unmatched_detections:
                dist = euclidean_distance(box_center(track.box), box_center(detections[idx].box))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx != -1 and best_dist <= self.max_distance:
                track.box = detections[best_idx].box
                track.disappeared = 0
                track.frames_seen += 1
                unmatched_detections.remove(best_idx)
                matched_track_ids.add(track_id)

        # Unmatched existing tracks: increment disappeared counter
        for track_id, track in self.tracks.items():
            if track_id not in matched_track_ids:
                track.disappeared += 1

        # Remaining detections become brand-new tracks
        for idx in unmatched_detections:
            tid = self._new_track_id()
            self.tracks[tid] = Track(track_id=tid, box=detections[idx].box)
            logger.debug("New track created: track_id=%d", tid)

        # Remove tracks that have disappeared too long
        exited: List[Track] = []
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id].disappeared > self.max_disappeared:
                exited.append(self.tracks.pop(track_id))
                logger.debug("Track %d removed after %d missed frames",
                             track_id, self.max_disappeared)

        active = list(self.tracks.values())
        return active, exited
