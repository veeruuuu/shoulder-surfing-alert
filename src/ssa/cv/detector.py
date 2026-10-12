"""MediaPipe Face Landmarker wrapper (§5.2, §6, §8 contract: detect())."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = REPO_ROOT / "models" / "face_landmarker.task"


@dataclass
class RawFace:
    """One detected face before tracking: bbox + raw landmarks."""
    bbox: tuple[int, int, int, int]   # x, y, w, h in pixels
    landmarks: np.ndarray             # (478, 3) normalized (x, y, z)


class Detector:
    """Wraps MediaPipe Face Landmarker. §8: detector.detect(frame_bgr) -> list[RawFace]."""

    def __init__(self, max_faces: int = 4, min_confidence: float = 0.5) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run scripts/download_models.py first."
            )
        base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> list[RawFace]:
        """Run face landmark detection on one BGR frame."""
        frame_rgb = frame_bgr[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        result = self._landmarker.detect(mp_image)

        h, w = frame_bgr.shape[:2]
        raw_faces: list[RawFace] = []
        for face_landmarks in result.face_landmarks:
            pts = np.array(
                [[lm.x, lm.y, lm.z] for lm in face_landmarks], dtype=np.float32
            )
            xs = pts[:, 0] * w
            ys = pts[:, 1] * h
            x_min, x_max = float(xs.min()), float(xs.max())
            y_min, y_max = float(ys.min()), float(ys.max())
            bbox = (
                int(max(0, x_min)),
                int(max(0, y_min)),
                int(x_max - x_min),
                int(y_max - y_min),
            )
            raw_faces.append(RawFace(bbox=bbox, landmarks=pts))
        return raw_faces

    def close(self) -> None:
        self._landmarker.close()
