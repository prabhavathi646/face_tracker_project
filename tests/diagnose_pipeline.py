"""Diagnostic: run one pipeline iteration manually and print timings."""
import sys
import time

sys.path.insert(0, ".")
import cv2

from src.config import Config
from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
from src.utils import prepare_face_crop

cfg = Config("config.json")

t0 = time.time()
det = FaceDetector(cfg)
print(f"detector load: {time.time() - t0:.1f}s")

t0 = time.time()
rec = FaceRecognizer(cfg)
print(f"recognizer load: {time.time() - t0:.1f}s")

cap = cv2.VideoCapture("data/sample_video.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 285)  # near a frame with 2 faces
ok, frame = cap.read()
cap.release()
assert ok

t0 = time.time()
detections = det.detect(frame)
dt = time.time() - t0
print(f"detect: {dt*1000:.0f}ms, {len(detections)} detections")

for d in detections:
    x1, y1, x2, y2 = [int(v) for v in d.box]
    tight = frame[y1:y2, x1:x2]
    margin = cfg.get("recognition", "crop_margin", default=0.4)
    crop = prepare_face_crop(frame, d.box, margin=margin, min_size=160)
    print(f"  crop shape: {crop.shape}")
    t0 = time.time()
    emb = rec.get_embedding(crop)
    dt = time.time() - t0
    print(f"  embedding: {dt*1000:.0f}ms, {'OK dim=' + str(len(emb)) if emb is not None else 'FAILED (None)'}")
    t0 = time.time()
    emb_tight = rec.get_embedding(tight)
    print(f"  tight-crop embedding: {'OK' if emb_tight is not None else 'FAILED'} "
          f"(control, {time.time() - t0:.2f}s)")

# Also test embedding on the full frame (as a sanity check)
t0 = time.time()
emb = rec.get_embedding(frame)
dt = time.time() - t0
print(f"full-frame embedding: {dt*1000:.0f}ms, {'OK' if emb is not None else 'FAILED'}")