"""Smoke test: load InsightFace buffalo_l and embed a frame."""
import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))
cap = cv2.VideoCapture("data/sample_video.mp4")
ok, frame = cap.read()
cap.release()
assert ok, "could not read sample video"
faces = app.get(frame)
print(f"faces_found={len(faces)}")
for f in faces:
    print("det_score:", round(float(f.det_score), 3),
          "embedding_dim:", len(f.normed_embedding))