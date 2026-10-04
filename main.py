"""Entry point for the Intelligent Face Tracker with Auto-Registration
and Visitor Counting.

Pipeline per frame:
    1. Read frame from VideoSource (file or RTSP).
    2. Every ``frame_skip`` frames, run YOLO face detection.
    3. Update the tracker with new/propagated detections.
    4. For each track missing a face_id (or due for re-check), crop the
       face, generate an ArcFace embedding, and match against known faces.
    5. If matched -> associate persistent face_id, log sighting.
       If unmatched -> auto-register a brand-new face_id.
    6. On first successful association -> log ENTRY event.
    7. When a track disappears for too long -> log EXIT event.
    8. Periodically report the running unique visitor count.

Run with: ``python main.py --config config.json``
"""

from __future__ import annotations

import argparse
import sys

import cv2

from src.config import Config, ConfigError
from src.database import Database
from src.detector import FaceDetector, DetectorLoadError
from src.event_logger import EventLogger, setup_logging
from src.recognizer import FaceRecognizer, RecognizerLoadError
from src.registration import FaceRegistrar
from src.tracker import FaceTracker
from src.utils import clamp_box, prepare_face_crop
from src.video_source import VideoSource, VideoSourceError
from src.visitor_counter import VisitorCounter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intelligent Face Tracker")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    return parser.parse_args()


def run(config_path: str) -> int:
    try:
        config = Config(config_path)
    except ConfigError as exc:
        print(f"[FATAL] Configuration error: {exc}", file=sys.stderr)
        return 1

    logger = setup_logging(config)

    try:
        db_path = config.resolve_path(config.get("database", "path", default="data/face_tracker.db"))
        db = Database(db_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fatal: could not initialize database: %s", exc)
        return 1

    try:
        detector = FaceDetector(config)
        recognizer = FaceRecognizer(config)
    except (DetectorLoadError, RecognizerLoadError) as exc:
        logger.error("Fatal model loading error: %s", exc)
        db.close()
        return 1

    tracker = FaceTracker(config)
    registrar = FaceRegistrar(config, db)
    event_logger = EventLogger(config, db)
    counter = VisitorCounter(db)

    try:
        video = VideoSource(config)
    except VideoSourceError as exc:
        logger.error("Fatal video source error: %s", exc)
        db.close()
        return 1

    frame_skip = max(1, config.get("detection", "frame_skip", default=2))
    display = config.get("processing", "display_window", default=True)
    min_face_px = config.get("processing", "min_face_size_px", default=30)
    crop_margin = config.get("recognition", "crop_margin", default=0.4)
    crop_min_size = config.get("recognition", "min_crop_size_px", default=160)

    frame_idx = 0
    last_detections = []
    logger.info("=== Face tracker started ===")

    try:
        while True:
            ok, frame = video.read()
            if not ok or frame is None:
                logger.info("No more frames / stream ended.")
                break

            height, width = frame.shape[:2]

            if frame_idx % frame_skip == 0:
                last_detections = detector.detect(frame)
            frame_idx += 1

            active_tracks, exited_tracks = tracker.update(last_detections)

            for track in exited_tracks:
                if track.face_id and track.entry_logged:
                    event_logger.log_exit(track.face_id, track.track_id, track.last_crop,
                                           track.best_similarity)

            for track in active_tracks:
                x1, y1, x2, y2 = clamp_box(track.box, width, height)
                if (x2 - x1) < min_face_px or (y2 - y1) < min_face_px:
                    continue
                # Pad + upscale: InsightFace needs face context to detect
                # faces inside tight YOLO crops (otherwise embeddings fail).
                face_crop = prepare_face_crop(frame, (x1, y1, x2, y2),
                                               margin=crop_margin,
                                               min_size=crop_min_size)
                if face_crop.size == 0:
                    continue
                track.last_crop = face_crop  # keep for the eventual EXIT image

                if track.face_id is None:
                    embedding = recognizer.get_embedding(face_crop)
                    if embedding is None:
                        continue
                    known = registrar.get_known_embeddings()
                    matched_id, similarity = recognizer.match(embedding, known)
                    if matched_id is not None:
                        track.face_id = matched_id
                        track.best_similarity = similarity
                        registrar.record_sighting(matched_id)
                        logger.info("Re-identified existing face %s (sim=%.3f) on track %d",
                                    matched_id, similarity, track.track_id)
                    else:
                        new_id = registrar.register_new_face(embedding, face_crop)
                        track.face_id = new_id
                        track.best_similarity = similarity
                        counter.register_new_visitor(new_id)

                if track.face_id and not track.entry_logged:
                    event_logger.log_entry(track.face_id, track.track_id, face_crop,
                                            track.best_similarity)
                    track.entry_logged = True

                if display:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    label = track.face_id or f"track-{track.track_id}"
                    cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

            if display:
                cv2.putText(frame, f"Unique visitors: {counter.count()}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.imshow("Intelligent Face Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User requested quit (pressed 'q').")
                    break

    except KeyboardInterrupt:
        logger.warning("Interrupted by user (KeyboardInterrupt).")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in main loop: %s", exc)
    finally:
        for track in tracker.tracks.values():
            if track.face_id and track.entry_logged:
                event_logger.log_exit(track.face_id, track.track_id, track.last_crop,
                                       track.best_similarity)

        video.release()
        if display:
            cv2.destroyAllWindows()
        entries, exits = db.get_event_counts()
        logger.info("=== Session summary: unique_visitors=%d entries=%d exits=%d ===",
                    counter.persisted_count(), entries, exits)
        print(f"\nUnique visitors counted: {counter.persisted_count()}")
        print(f"Total ENTRY events: {entries}")
        print(f"Total EXIT events: {exits}")
        db.close()

    return 0


if __name__ == "__main__":
    args = parse_args()
    sys.exit(run(args.config))
