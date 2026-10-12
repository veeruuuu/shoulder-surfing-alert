"""§14: distance.py unit tests — ratio and distance math, preset thresholds."""
from __future__ import annotations

import numpy as np

from ssa.cv import distance


def _make_landmarks(right_eye_px, left_eye_px, w, h):
    landmarks = np.zeros((478, 3), dtype=np.float32)
    landmarks[33] = [right_eye_px[0] / w, right_eye_px[1] / h, 0.0]
    landmarks[133] = [right_eye_px[0] / w, right_eye_px[1] / h, 0.0]
    landmarks[362] = [left_eye_px[0] / w, left_eye_px[1] / h, 0.0]
    landmarks[263] = [left_eye_px[0] / w, left_eye_px[1] / h, 0.0]
    return landmarks


def test_iod_px_matches_pixel_distance():
    w, h = 640, 480
    landmarks = _make_landmarks((200, 240), (260, 240), w, h)
    iod_px, ratio, est_cm = distance.compute(landmarks, (h, w), owner_iod_px=None, ref_cm=50)
    assert abs(iod_px - 60.0) < 1e-3
    assert ratio == 0.0
    assert est_cm == float("inf")


def test_ratio_and_distance_with_calibration():
    w, h = 640, 480
    landmarks = _make_landmarks((200, 240), (260, 240), w, h)  # iod_px = 60
    iod_px, ratio, est_cm = distance.compute(landmarks, (h, w), owner_iod_px=60.0, ref_cm=50)
    assert abs(ratio - 1.0) < 1e-6
    assert abs(est_cm - 50.0) < 1e-6


def test_closer_face_gives_higher_ratio_and_lower_distance():
    w, h = 640, 480
    close_landmarks = _make_landmarks((180, 240), (300, 240), w, h)  # iod_px = 120
    iod_px, ratio, est_cm = distance.compute(close_landmarks, (h, w), owner_iod_px=60.0, ref_cm=50)
    assert ratio == 2.0
    assert est_cm == 25.0


def test_preset_thresholds_from_config():
    from ssa.config import load_config
    presets = load_config()["distance"]["presets"]
    assert presets == {"close": 0.60, "medium": 0.40, "wide": 0.25}