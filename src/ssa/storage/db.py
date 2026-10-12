"""SQLite access layer (§10). CRUD for trusted_faces, intruder_events, calibration."""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = REPO_ROOT / "data" / "app.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trusted_faces (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  embedding BLOB NOT NULL,
  is_owner INTEGER DEFAULT 0,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS intruder_events (
  id INTEGER PRIMARY KEY,
  started_at REAL NOT NULL,
  ended_at REAL,
  duration_s REAL,
  max_ratio REAL,
  min_distance_cm REAL,
  yaw REAL,
  pitch REAL,
  snapshot_path TEXT
);

CREATE TABLE IF NOT EXISTS calibration (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  owner_iod_px REAL NOT NULL,
  ref_distance_cm REAL NOT NULL,
  calibrated_at REAL NOT NULL
);
"""


def init_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)


@contextmanager
def _connect(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def add_trusted_face(name: str, embedding: np.ndarray, is_owner: bool = False, db_path: Path = DB_PATH) -> int:
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO trusted_faces (name, embedding, is_owner, created_at) VALUES (?, ?, ?, ?)",
            (name, embedding.astype(np.float32).tobytes(), int(is_owner), time.time()),
        )
        return cur.lastrowid


def list_trusted_faces(db_path: Path = DB_PATH) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, name, embedding, is_owner, created_at FROM trusted_faces"
        ).fetchall()
    return [
        {
            "id": r[0],
            "name": r[1],
            "embedding": np.frombuffer(r[2], dtype=np.float32),
            "is_owner": bool(r[3]),
            "created_at": r[4],
        }
        for r in rows
    ]


def delete_trusted_face(face_id: int, db_path: Path = DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM trusted_faces WHERE id = ?", (face_id,))


def add_intruder_event(
    started_at: float, ended_at: Optional[float], max_ratio: float,
    min_distance_cm: float, yaw: float, pitch: float, snapshot_path: str,
    db_path: Path = DB_PATH,
) -> int:
    duration_s = (ended_at - started_at) if ended_at is not None else None
    with _connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO intruder_events
               (started_at, ended_at, duration_s, max_ratio, min_distance_cm, yaw, pitch, snapshot_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (started_at, ended_at, duration_s, max_ratio, min_distance_cm, yaw, pitch, snapshot_path),
        )
        return cur.lastrowid


def list_intruder_events(db_path: Path = DB_PATH) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """SELECT id, started_at, ended_at, duration_s, max_ratio, min_distance_cm,
                      yaw, pitch, snapshot_path FROM intruder_events ORDER BY started_at DESC"""
        ).fetchall()
    cols = ["id", "started_at", "ended_at", "duration_s", "max_ratio",
            "min_distance_cm", "yaw", "pitch", "snapshot_path"]
    return [dict(zip(cols, r)) for r in rows]


def delete_intruder_event(event_id: int, db_path: Path = DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM intruder_events WHERE id = ?", (event_id,))


def set_calibration(owner_iod_px: float, ref_distance_cm: float, db_path: Path = DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """INSERT INTO calibration (id, owner_iod_px, ref_distance_cm, calibrated_at)
               VALUES (1, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 owner_iod_px = excluded.owner_iod_px,
                 ref_distance_cm = excluded.ref_distance_cm,
                 calibrated_at = excluded.calibrated_at""",
            (owner_iod_px, ref_distance_cm, time.time()),
        )


def get_calibration(db_path: Path = DB_PATH) -> Optional[dict]:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT owner_iod_px, ref_distance_cm, calibrated_at FROM calibration WHERE id = 1"
        ).fetchone()
    if row is None:
        return None
    return {"owner_iod_px": row[0], "ref_distance_cm": row[1], "calibrated_at": row[2]}
