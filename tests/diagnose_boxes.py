"""Compare YOLO boxes vs InsightFace's own detections on one frame."""
import sys

sys.path.insert(0, ".")
import cv2

from src.config import Config
from src.detector import FaceDetector
from src.recognizer import FaceRecognizer

cfg = Config("config.json")
det = FaceDetector(cfg)
rec = FaceRecognizer(cfg)

cap = cv2.VideoCapture("data/sample_video.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 285)
ok, frame = cap.read()
cap.release()
assert ok

yolo_boxes = [d.box for d in det.detect(frame)]
print("YOLO boxes:", [[round(v) for v in b] for b in yolo_boxes])

faces = rec._app.get(frame)
print(f"InsightFace full-frame faces: {len(faces)}")
for f in faces:
    bbox = f.bbox.round().astype(int).tolist()
    print("  IF box:", bbox, "score:", round(float(f.det_score), 3),
          "emb_dim:", len(f.normed_embedding))

# Try each YOLO box padded 50% — does InsightFace see a face there?
for b in yolo_boxes:
    x1, y1, x2, y2 = b
    bw, bh = (x2 - x1) * 0.5, (y2 - y1) * 0.5
    px1 = max(0, int(x1 - bw)); py1 = max(0, int(y1 - bh))
    px2 = min(frame.shape[1], int(x2 + bw)); py2 = min(frame.shape[0], int(y2 + bh))
    crop = frame[py1:py2, px1:px2]
    sub = rec._app.get(crop)
    print(f"YOLO box {[round(v) for v in b]} padded crop {crop.shape} -> "
          f"{len(sub)} face(s) by InsightFace")