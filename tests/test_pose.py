"""§14: pose.py unit tests — synthetic landmarks with known rotation should
recover approximately the same yaw/pitch via solvePnP."""
from __future__ import annotations

import cv2
import numpy as np

from ssa.cv.pose import _LANDMARK_IDX, _MODEL_POINTS, estimate, PoseSmoother


def _synthetic_landmarks_for_angles(yaw_deg, pitch_deg, roll_deg, w=640, h=480):
    yaw, pitch, roll = np.radians([yaw_deg, pitch_deg, roll_deg])

    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    rmat = rz @ ry @ rx

    rvec, _ = cv2.Rodrigues(rmat)
    tvec = np.array([[0.0], [0.0], [600.0]])

    focal_length = float(w)
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64
    )
    dist_coeffs = np.zeros((4, 1))

    image_points, _ = cv2.projectPoints(_MODEL_POINTS, rvec, tvec, camera_matrix, dist_coeffs)
    image_points = image_points.reshape(-1, 2)

    landmarks = np.zeros((478, 3), dtype=np.float32)
    names = ["nose_tip", "chin", "left_eye_outer", "right_eye_outer", "left_mouth", "right_mouth"]
    for name, pt in zip(names, image_points):
        idx = _LANDMARK_IDX[name]
        landmarks[idx] = [pt[0] / w, pt[1] / h, 0.0]
    return landmarks


def test_frontal_face_gives_near_zero_angles():
    landmarks = _synthetic_landmarks_for_angles(0, 0, 0)
    yaw, pitch, roll = estimate(landmarks, (480, 640))
    assert abs(yaw) < 2.0
    assert abs(pitch) < 2.0


def test_turned_head_recovers_yaw():
    landmarks = _synthetic_landmarks_for_angles(yaw_deg=30, pitch_deg=0, roll_deg=0)
    yaw, pitch, roll = estimate(landmarks, (480, 640))
    assert abs(yaw - 30) < 5.0


def test_tilted_head_recovers_pitch():
    landmarks = _synthetic_landmarks_for_angles(yaw_deg=0, pitch_deg=20, roll_deg=0)
    yaw, pitch, roll = estimate(landmarks, (480, 640))
    assert abs(pitch - 20) < 5.0


def test_pose_smoother_ema():
    smoother = PoseSmoother(alpha=0.4)
    y1, p1, r1 = smoother.smooth(track_id=0, yaw=0.0, pitch=0.0, roll=0.0)
    assert (y1, p1, r1) == (0.0, 0.0, 0.0)
    y2, p2, r2 = smoother.smooth(track_id=0, yaw=10.0, pitch=0.0, roll=0.0)
    assert abs(y2 - 4.0) < 1e-6

def test_wrap_pitch_corrects_near_180_to_near_0():
    from ssa.cv.pose import _wrap_pitch
    assert abs(_wrap_pitch(179.0) - (-1.0)) < 1e-6
    assert abs(_wrap_pitch(175.0) - (-5.0)) < 1e-6
    assert abs(_wrap_pitch(-179.0) - 1.0) < 1e-6


def test_wrap_pitch_leaves_normal_range_unchanged():
    from ssa.cv.pose import _wrap_pitch
    assert _wrap_pitch(0.0) == 0.0
    assert _wrap_pitch(45.0) == 45.0
    assert _wrap_pitch(-45.0) == -45.0
