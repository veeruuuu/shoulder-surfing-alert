"""Manual check for recognizer.py (§21.7). Not a pytest test — needs a live
camera. Run: python tests/manual_check_recognizer.py
Expect: matching your own face to itself gives identity=owner, and a
deliberately dissimilar embedding is classified as unknown."""
from __future__ import annotations

import numpy as np

from ssa.cv.capture import Capture
from ssa.cv.detector import Detector
from ssa.cv.recognizer import GalleryEntry, Recognizer


def main() -> None:
    with Capture(index=0, width=640, height=480) as cap:
        detector = Detector(max_faces=1, min_confidence=0.5)
        recognizer = Recognizer()
        try:
            print("Look at the camera. Searching for a face...")
            frame = None
            raw_faces = []
            for _ in range(60):
                f = cap.read()
                if f is None:
                    continue
                faces = detector.detect(f)
                if faces:
                    frame = f
                    raw_faces = faces
                    break

            if frame is None or not raw_faces:
                print("No face detected. Make sure you're in frame and well lit.")
                return

            emb = recognizer.embed(frame, raw_faces[0].landmarks)
            print(f"Embedding shape: {emb.shape}")

            gallery = [GalleryEntry(name="me", is_owner=True, embedding=emb)]
            identity, name = recognizer.match(emb, gallery)
            print(f"Self-match -> identity={identity}, name={name} (expect owner, None)")

            fake_emb = np.ones_like(emb) * -1
            identity2, name2 = recognizer.match(fake_emb, gallery)
            print(f"Dissimilar-match -> identity={identity2}, name={name2} (expect unknown, None)")
        finally:
            detector.close()


if __name__ == "__main__":
    main()
    