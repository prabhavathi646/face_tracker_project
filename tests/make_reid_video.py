"""Build a leave-and-return test video to prove live re-identification.

Segment A: frames 285-350 of the sample video (people visible).
Gap: 60 black frames (no detections -> tracks retire -> EXIT events).
Segment B: exact replay of segment A (same people -> must be
re-identified to their EXISTING face ids, with no new registrations).

Expected after running main.py on this video:
  - unique visitor count == people seen in segment A (NOT doubled)
  - "Re-identified existing face ..." lines in events.log
  - more ENTRY events than face rows (re-entry allowed, re-registration not)
"""
import os

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC = os.path.join(ROOT, "data", "sample_video.mp4")
DST = os.path.join(ROOT, "data", "reid_sample.mp4")
START, END, GAP = 285, 350, 60


def read_segment():
    cap = cv2.VideoCapture(SRC)
    cap.set(cv2.CAP_PROP_POS_FRAMES, START)
    frames = []
    for _ in range(END - START):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def main() -> None:
    segment = read_segment()
    if not segment:
        raise SystemExit("could not read source segment")
    h, w, _ = segment[0].shape
    out = cv2.VideoWriter(DST, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    for frame in segment:
        out.write(frame)
    black = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(GAP):
        out.write(black)  # long enough for max_disappeared_frames to retire tracks
    for frame in segment:
        out.write(frame)
    out.release()
    print(f"Wrote {DST}: {len(segment)}x2 frames + {GAP} black")


if __name__ == "__main__":
    main()