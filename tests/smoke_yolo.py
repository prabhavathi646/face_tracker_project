"""Smoke test: load the YOLO face model and detect faces on frame 0."""
import cv2
from ultralytics import YOLO

model = YOLO("models/yolov8n-face.pt")
cap = cv2.VideoCapture("data/sample_video.mp4")
ok, frame = cap.read()
cap.release()
assert ok, "could not read sample video"
results = model.predict(frame, conf=0.4, verbose=False)
boxes = results[0].boxes
print(f"frame_ok={ok} detections={len(boxes)}")
for xyxy in boxes.xyxy.tolist():
    print("box:", [round(v) for v in xyxy])