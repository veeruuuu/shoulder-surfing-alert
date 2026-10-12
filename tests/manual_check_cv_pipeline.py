"""Manual check for capture + detector + tracker (§21.7). Not a pytest test —
opens a live camera window. Run: python tests/manual_check_cv_pipeline.py
Expect: your face in a green box with a track ID that stays the same number
as you move. With a second face in frame, it should get its own box/ID.
Press 'q' to quit."""
from __future__ import annotations

import cv2

from ssa.cv.capture import Capture
from ssa.cv.detector import Detector
from ssa.cv.tracker import Tracker


def main() -> None:
    with Capture(index=0, width=640, height=480) as cap:
        detector = Detector(max_faces=4, min_confidence=0.5)
        tracker = Tracker()
        try:
            while True:
                frame = cap.read()
                if frame is None:
                    print("Frame read failed")
                    break

                raw_faces = detector.detect(frame)
                tracked = tracker.update(raw_faces)

                for t in tracked:
                    x, y, w, h = t.bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        frame, f"id={t.track_id}", (x, max(0, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )

                cv2.imshow("SSA manual check - press q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            detector.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()