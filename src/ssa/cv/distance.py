"""Distance estimation from inter-ocular distance (§4.1, §8 contract: compute()).
Rough estimate only; label "approx" in UI/report per §4.1."""
from __future__ import annotations

import numpy as np

# Eye landmark indices, consistent with the §8 5-point SFace alignment note:
# right eye = mean(33, 133), left eye = mean(362, 263).
_RIGHT_EYE_IDX = (33, 133)
_LEFT_EYE_IDX = (362, 263)


def compute(
    landmarks: np.ndarray,
    frame_shape: tuple[int, int],
    owner_iod_px: float | None,
    ref_cm: float,
) -> tuple[float, float, float]:
    """§8 contract: distance.compute(landmarks, frame_shape, owner_iod_px, ref_cm)
    -> (iod_px, ratio, est_cm).

    owner_iod_px: calibrated owner inter-ocular distance in pixels (None/0 if
    not yet calibrated — ratio and est_cm are then meaningless, returned as
    0.0 and inf so callers can detect the uncalibrated case).
    ref_cm: owner_ref_distance_cm from config (§9, default 50).
    """
    h, w = frame_shape[0], frame_shape[1]

    right_eye = np.mean([landmarks[i][:2] for i in _RIGHT_EYE_IDX], axis=0) * [w, h]
    left_eye = np.mean([landmarks[i][:2] for i in _LEFT_EYE_IDX], axis=0) * [w, h]

    iod_px = float(np.hypot(*(right_eye - left_eye)))

    if not owner_iod_px:
        return iod_px, 0.0, float("inf")

    ratio = iod_px / owner_iod_px
    est_cm = ref_cm / ratio if ratio > 0 else float("inf")
    return iod_px, ratio, est_cm