"""OS-specific code only (§5.5, §21.6). Windows-only tested; macOS branches
kept per spec (§2, §5.5) but not exercised in this work mode."""
from __future__ import annotations

import ctypes
import platform
import subprocess

import cv2


def get_camera_backend() -> int:
    """Return the OpenCV capture backend for the current OS."""
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW
    elif system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    else:
        return cv2.CAP_ANY


def lock_screen() -> None:
    """Lock the OS screen (§5.5, used for away_action: lock)."""
    system = platform.system()
    if system == "Windows":
        ctypes.windll.user32.LockWorkStation()
    elif system == "Darwin":
        subprocess.run(["pmset", "displaysleepnow"], check=False)
    else:
        raise NotImplementedError(f"Screen lock not implemented for {system}")


def set_dpi_aware() -> None:
    """Make this process DPI-aware on Windows so Tkinter reports true physical
    screen pixels, matching mss's capture resolution (§5.4). Without this,
    Windows display scaling (125%/150%) makes Tkinter windows undersize
    themselves, leaving the overlay covering only part of the screen.
    Must be called before any Tk root is created."""
    if platform.system() == "Windows":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI aware
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()  # fallback, older Windows
            except Exception:
                pass
