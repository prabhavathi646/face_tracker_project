"""Generate a synthetic sample video for pipeline smoke-testing.

Draws moving circles (stand-ins for faces) that enter from the left,
cross the frame, and exit on the right. Useful for validating the
tracking/event plumbing WITHOUT ML models. For real face recognition
tests, point ``config.json`` at a video containing actual faces.

Usage: python tests/make_sample_video.py
"""
import os

import cv2
import numpy as np

OUTPUT = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic.mp4")


def main() -> None:
    width, height, fps, frames = 640, 480, 25, 150
    out = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for i in range(frames):
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
        # Two "visitors": one crossing left->right, one right->left.
        x1 = int(-80 + (width + 160) * i / frames)
        y1 = 240
        cv2.circle(frame, (x1, y1), 40, (200, 180, 160), -1)
        cv2.circle(frame, (x1 - 12, y1 - 8), 6, (40, 40, 40), -1)
        cv2.circle(frame, (x1 + 12, y1 - 8), 6, (40, 40, 40), -1)

        x2 = int(width + 80 - (width + 160) * i / frames)
        cv2.circle(frame, (x2, 120), 35, (170, 160, 200), -1)

        out.write(frame)

    out.release()
    print(f"Wrote {OUTPUT} ({frames} frames @ {fps} FPS)")


if __name__ == "__main__":
    main()