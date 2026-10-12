"""Shared dataclasses (§8) and thread-safe AppState."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np

Identity = Literal["owner", "trusted", "unknown"]
AlertState = Literal["SAFE", "ALERT", "AWAY", "PAUSED"]


@dataclass
class FaceObs:
    track_id: int
    bbox: tuple[int, int, int, int]      # x, y, w, h in pixels
    landmarks: np.ndarray                # (478, 3) normalized
    iod_px: float
    ratio: float                         # iod_px / owner_iod_px
    est_distance_cm: float
    yaw: float                           # degrees, smoothed
    pitch: float
    roll: float
    identity: Identity
    trusted_name: Optional[str]
    near: bool
    looking: bool
    is_peeker: bool


@dataclass
class FrameResult:
    timestamp: float
    faces: list[FaceObs]
    state: AlertState
    fps: float
    owner_present: bool


@dataclass
class PeekerEvent:
    started_at: float
    ended_at: Optional[float]
    max_ratio: float
    min_distance_cm: float
    yaw: float
    pitch: float
    snapshot_path: str


class AppState:
    """Thread-safe shared state between the CV thread, main thread, and web thread.

    §5.1 says shared state goes through "one thread-safe AppState object"
    without specifying its shape, so this stays a lock-guarded holder for
    whatever the dashboard/overlay need to read or toggle, extended as each
    component needs it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_frame_result: Optional[FrameResult] = None
        self._paused: bool = False
        self._calibrated: bool = False
        self._camera_active: bool = False
        self._shutting_down: bool = False
        self._preview_jpeg: Optional[bytes] = None
        self._camera_error: Optional[str] = None

        self._cv_task_request: Optional[dict[str, Any]] = None
        self._cv_task_result: Optional[dict[str, Any]] = None

    def set_frame_result(self, result: FrameResult) -> None:
        with self._lock:
            self._latest_frame_result = result

    def get_frame_result(self) -> Optional[FrameResult]:
        with self._lock:
            return self._latest_frame_result

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._paused = paused

    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    def set_calibrated(self, calibrated: bool) -> None:
        with self._lock:
            self._calibrated = calibrated

    def is_calibrated(self) -> bool:
        with self._lock:
            return self._calibrated

    def set_camera_active(self, active: bool) -> None:
        with self._lock:
            self._camera_active = active

    def is_camera_active(self) -> bool:
        with self._lock:
            return self._camera_active

    def request_shutdown(self) -> None:
        with self._lock:
            self._shutting_down = True

    def is_shutting_down(self) -> bool:
        with self._lock:
            return self._shutting_down

    def set_preview_jpeg(self, data: bytes) -> None:
        with self._lock:
            self._preview_jpeg = data

    def get_preview_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._preview_jpeg

    def set_camera_error(self, message: Optional[str]) -> None:
        """Set a user-facing camera error (permission denied, camera busy,
        no camera found, etc.), or None once the camera is working."""
        with self._lock:
            self._camera_error = message

    def get_camera_error(self) -> Optional[str]:
        with self._lock:
            return self._camera_error

    def request_cv_task(self, kind: str, **params: Any) -> None:
        with self._lock:
            self._cv_task_request = {"kind": kind, **params}
            self._cv_task_result = None

    def pop_cv_task_request(self) -> Optional[dict[str, Any]]:
        with self._lock:
            req = self._cv_task_request
            self._cv_task_request = None
            return req

    def set_cv_task_result(self, result: dict[str, Any]) -> None:
        with self._lock:
            self._cv_task_result = result

    def pop_cv_task_result(self) -> Optional[dict[str, Any]]:
        with self._lock:
            result = self._cv_task_result
            self._cv_task_result = None
            return result
