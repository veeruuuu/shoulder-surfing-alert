"""SFace face embeddings + trusted/owner matching (§4.3, §6, §8 contract:
embed(), match()). 5-point alignment per §8 note."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ssa.state import Identity

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = REPO_ROOT / "models" / "face_recognition_sface_2021dec.onnx"

# §8 5-point alignment note, consistent with pose.py / distance.py landmark
# choices: right eye = mean(33, 133), left eye = mean(362, 263), nose tip = 1,
# mouth corners = 61 (right) and 291 (left).
_RIGHT_EYE_IDX = (33, 133)
_LEFT_EYE_IDX = (362, 263)
_NOSE_TIP_IDX = 1
_RIGHT_MOUTH_IDX = 61
_LEFT_MOUTH_IDX = 291


@dataclass
class GalleryEntry:
    """One stored trusted/owner face, as loaded from storage.db."""
    name: str
    is_owner: bool
    embedding: np.ndarray


class Recognizer:
    """Wraps OpenCV FaceRecognizerSF. §8: embed(), match()."""

    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run scripts/download_models.py first."
            )
        self._sface = cv2.FaceRecognizerSF.create(str(MODEL_PATH), "")

    def embed(self, frame_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """§8 contract: recognizer.embed(frame_bgr, landmarks) -> np.ndarray.

        landmarks: (478, 3) normalized (x, y, z) from the detector, for ONE face.
        """
        h, w = frame_bgr.shape[:2]

        def px(idx: int) -> tuple[float, float]:
            return float(landmarks[idx][0] * w), float(landmarks[idx][1] * h)

        def mean_px(idx_a: int, idx_b: int) -> tuple[float, float]:
            ax, ay = px(idx_a)
            bx, by = px(idx_b)
            return (ax + bx) / 2.0, (ay + by) / 2.0

        right_eye = mean_px(*_RIGHT_EYE_IDX)
        left_eye = mean_px(*_LEFT_EYE_IDX)
        nose_tip = px(_NOSE_TIP_IDX)
        right_mouth = px(_RIGHT_MOUTH_IDX)
        left_mouth = px(_LEFT_MOUTH_IDX)

        xs = [right_eye[0], left_eye[0], nose_tip[0], right_mouth[0], left_mouth[0]]
        ys = [right_eye[1], left_eye[1], nose_tip[1], right_mouth[1], left_mouth[1]]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        pad_x = (x_max - x_min) * 0.6
        pad_y = (y_max - y_min) * 0.6
        bx = max(0.0, x_min - pad_x)
        by = max(0.0, y_min - pad_y)
        bw = min(w - bx, (x_max - x_min) + 2 * pad_x)
        bh = min(h - by, (y_max - y_min) + 2 * pad_y)

        face_info = np.array(
            [[
                bx, by, bw, bh,
                right_eye[0], right_eye[1],
                left_eye[0], left_eye[1],
                nose_tip[0], nose_tip[1],
                right_mouth[0], right_mouth[1],
                left_mouth[0], left_mouth[1],
            ]],
            dtype=np.float32,
        )

        aligned = self._sface.alignCrop(frame_bgr, face_info)
        feature = self._sface.feature(aligned)
        return feature.flatten().astype(np.float32)

    def match(
        self, emb: np.ndarray, gallery: list[GalleryEntry], threshold: float = 0.363
    ) -> tuple[Identity, Optional[str]]:
        """§8 contract: recognizer.match(emb, gallery) -> (Identity, name|None).
        §4.3: cosine similarity, default threshold 0.363."""
        best_score = -1.0
        best_entry: Optional[GalleryEntry] = None

        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        for entry in gallery:
            cand = entry.embedding / (np.linalg.norm(entry.embedding) + 1e-8)
            score = float(np.dot(emb_norm, cand))
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is not None and best_score >= threshold:
            if best_entry.is_owner:
                return "owner", None
            return "trusted", best_entry.name

        return "unknown", None