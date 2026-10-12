"""Camera capture (§5.3 step 1, §6). OS-specific backend from platform_utils."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ssa.platform_utils import get_camera_backend


class CameraAccessError(RuntimeError):
    """Raised when the camera cannot be opened. message is user-facing
    (§15 B7: permission checks) -- distinguishes likely causes where
    possible, since OpenCV itself gives no structured error code on Windows."""


class Capture:
    """Thin wrapper around cv2.VideoCapture with the configured resolution."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        backend = get_camera_backend()
        self._cap = cv2.VideoCapture(self._index, backend)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if not self._cap.isOpened():
            self._cap = None
            raise CameraAccessError(
                f"Could not open camera index {self._index}. This usually means "
                "either: (1) Windows camera permission is denied for this app "
                "(check Settings > Privacy & security > Camera), (2) another "
                "app (Zoom, Teams, browser, etc.) is already using the camera "
                "exclusively, or (3) no camera exists at that index. Close other "
                "camera apps, check camera permissions, and try again."
            )

    def read(self) -> Optional[np.ndarray]:
        """Grab one BGR frame, or None if the read failed."""
        if self._cap is None:
            raise RuntimeError("Capture.open() must be called before read()")
        ok, frame = self._cap.read()
        if not ok:
            return None
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Capture":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()
