"""Snapshot saving and retention cleanup (§10, §13)."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOTS_DIR = REPO_ROOT / "data" / "snapshots"


def save_snapshot(frame_bgr: np.ndarray, event_started_at: float) -> str:
    """Save one intruder snapshot. Returns the relative path stored in the DB."""
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{int(event_started_at * 1000)}.jpg"
    dest = SNAPSHOTS_DIR / filename
    cv2.imwrite(str(dest), frame_bgr)
    return f"data/snapshots/{filename}"


def cleanup_old_snapshots(retention_days: int) -> int:
    """Delete snapshot files older than retention_days. Returns count deleted."""
    if not SNAPSHOTS_DIR.exists():
        return 0
    cutoff = time.time() - retention_days * 86400
    deleted = 0
    for f in SNAPSHOTS_DIR.glob("*.jpg"):
        if f.stat().st_mtime < cutoff:
            f.unlink()
            deleted += 1
    return deleted
