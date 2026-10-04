"""Scan sample video frames and report face detections over time."""
import cv2
from ultralytics import YOLO

model = YOLO("models/yolov8n-face.pt")
cap = cv2.VideoCapture("data/sample_video.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("total frames:", total)
for idx in range(0, total, max(1, total // 10)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        continue
    r = model.predict(frame, conf=0.3, verbose=False)
    n = len(r[0].boxes)
    sizes = []
    for x1, y1, x2, y2 in r[0].boxes.xyxy.tolist():
        sizes.append((int(x2 - x1), int(y2 - y1)))
    print(f"frame {idx}: {n} faces {sizes[:6]}")
cap.release()