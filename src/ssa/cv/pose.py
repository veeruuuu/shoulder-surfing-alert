"""Head pose estimation via cv2.solvePnP (§4.2, §6, §8 contract: estimate()).
Head orientation as a proxy for gaze, not true eye tracking (§19)."""
from __future__ import annotations

import math

import cv2
import numpy as np

_LANDMARK_IDX = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)


def _rotation_matrix_to_euler_deg(rmat: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(rmat[2, 1], rmat[2, 2])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = 0.0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def _wrap_pitch(pitch_deg: float) -> float:
    """Correct the ~180-degree pitch offset this model/decomposition produces
    for a frontal face (object-space Z axis points toward the camera, opposite
    OpenCV's forward-Z convention). Confirmed empirically on 2026-10-11: live
    frontal faces read pitch ~175-179 degrees instead of ~0."""
    if pitch_deg > 90:
        return pitch_deg - 180
    if pitch_deg < -90:
        return pitch_deg + 180
    return pitch_deg


def estimate(landmarks: np.ndarray, frame_shape: tuple[int, int]) -> tuple[float, float, float]:
    """§8 contract: pose.estimate(landmarks, frame_shape) -> (yaw, pitch, roll).
    Returns raw (unsmoothed) yaw, pitch, roll in degrees, pitch wrapped to be
    near 0 for a frontal face (see _wrap_pitch)."""
    h, w = frame_shape[0], frame_shape[1]

    image_points = np.array(
        [
            (landmarks[_LANDMARK_IDX["nose_tip"]][0] * w, landmarks[_LANDMARK_IDX["nose_tip"]][1] * h),
            (landmarks[_LANDMARK_IDX["chin"]][0] * w, landmarks[_LANDMARK_IDX["chin"]][1] * h),
            (landmarks[_LANDMARK_IDX["left_eye_outer"]][0] * w, landmarks[_LANDMARK_IDX["left_eye_outer"]][1] * h),
            (landmarks[_LANDMARK_IDX["right_eye_outer"]][0] * w, landmarks[_LANDMARK_IDX["right_eye_outer"]][1] * h),
            (landmarks[_LANDMARK_IDX["left_mouth"]][0] * w, landmarks[_LANDMARK_IDX["left_mouth"]][1] * h),
            (landmarks[_LANDMARK_IDX["right_mouth"]][0] * w, landmarks[_LANDMARK_IDX["right_mouth"]][1] * h),
        ],
        dtype=np.float64,
    )

    focal_length = float(w)
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, _tvec = cv2.solvePnP(
        _MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = _rotation_matrix_to_euler_deg(rmat)
    pitch = _wrap_pitch(pitch)
    return yaw, pitch, roll


class PoseSmoother:
    """Per-track exponential moving average smoothing (§4.2: alpha=0.4),
    applied by the pipeline on top of estimate()'s raw output."""

    def __init__(self, alpha: float = 0.4) -> None:
        self._alpha = alpha
        self._state: dict[int, tuple[float, float, float]] = {}

    def smooth(self, track_id: int, yaw: float, pitch: float, roll: float) -> tuple[float, float, float]:
        prev = self._state.get(track_id)
        if prev is None:
            smoothed = (yaw, pitch, roll)
        else:
            a = self._alpha
            smoothed = (
                a * yaw + (1 - a) * prev[0],
                a * pitch + (1 - a) * prev[1],
                a * roll + (1 - a) * prev[2],
            )
        self._state[track_id] = smoothed
        return smoothed

    def forget(self, track_id: int) -> None:
        self._state.pop(track_id, None)
