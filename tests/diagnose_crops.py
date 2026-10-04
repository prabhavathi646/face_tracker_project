"""Isolate which crop variant makes InsightFace succeed/fail."""
import sys

sys.path.insert(0, ".")
import cv2
import numpy as np

from src.config import Config
from src.recognizer import FaceRecognizer

cfg = Config("config.json")
rec = FaceRecognizer(cfg)

cap = cv2.VideoCapture("data/sample_video.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 285)
ok, frame = cap.read()
cap.release()
assert ok

# Same boxes verified in diagnose_boxes.py
boxes = [(196, 190, 243, 249), (486, 125, 546, 198)]

for box in boxes:
    x1, y1, x2, y2 = box
    print(f"box {box}:")
    for margin in (0.0, 0.2, 0.3, 0.5):
        bw, bh = (x2 - x1) * margin, (y2 - y1) * margin
        px1 = max(0, int(x1 - bw)); py1 = max(0, int(y1 - bh))
        px2 = min(frame.shape[1], int(x2 + bw)); py2 = min(frame.shape[0], int(y2 + bh))
        crop = frame[py1:py2, px1:px2]
        emb = rec.get_embedding(crop)
        upscaled = cv2.resize(crop, (160, int(160 * crop.shape[0] / crop.shape[1])),
                              interpolation=cv2.INTER_CUBIC) if crop.shape[1] < 160 else crop
        emb_up = rec.get_embedding(upscaled)
        print(f"  margin={margin}: native {crop.shape[:2]} -> {'OK' if emb is not None else 'FAIL'}, "
              f"upscaled {upscaled.shape[:2]} -> {'OK' if emb_up is not None else 'FAIL'}")