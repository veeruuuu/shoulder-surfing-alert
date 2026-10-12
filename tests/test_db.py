"""§14: db.py unit tests — CRUD for trusted faces and events."""
from __future__ import annotations

import numpy as np
import pytest

from ssa.storage import db


@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "test_app.db"
    db.init_db(db_path)
    return db_path


def test_add_and_list_trusted_face(tmp_db):
    emb = np.random.rand(128).astype(np.float32)
    face_id = db.add_trusted_face("Alice", emb, is_owner=False, db_path=tmp_db)
    faces = db.list_trusted_faces(db_path=tmp_db)
    assert len(faces) == 1
    assert faces[0]["id"] == face_id
    assert faces[0]["name"] == "Alice"
    assert faces[0]["is_owner"] is False
    np.testing.assert_array_almost_equal(faces[0]["embedding"], emb)


def test_delete_trusted_face(tmp_db):
    emb = np.random.rand(128).astype(np.float32)
    face_id = db.add_trusted_face("Bob", emb, db_path=tmp_db)
    db.delete_trusted_face(face_id, db_path=tmp_db)
    assert db.list_trusted_faces(db_path=tmp_db) == []


def test_add_and_list_intruder_event(tmp_db):
    event_id = db.add_intruder_event(
        started_at=100.0, ended_at=105.0, max_ratio=0.8,
        min_distance_cm=30.0, yaw=10.0, pitch=5.0,
        snapshot_path="data/snapshots/100000.jpg", db_path=tmp_db,
    )
    events = db.list_intruder_events(db_path=tmp_db)
    assert len(events) == 1
    assert events[0]["id"] == event_id
    assert events[0]["duration_s"] == 5.0


def test_delete_intruder_event(tmp_db):
    event_id = db.add_intruder_event(
        started_at=1.0, ended_at=2.0, max_ratio=0.5,
        min_distance_cm=40.0, yaw=0.0, pitch=0.0,
        snapshot_path="x.jpg", db_path=tmp_db,
    )
    db.delete_intruder_event(event_id, db_path=tmp_db)
    assert db.list_intruder_events(db_path=tmp_db) == []


def test_calibration_set_and_get(tmp_db):
    assert db.get_calibration(db_path=tmp_db) is None
    db.set_calibration(owner_iod_px=60.0, ref_distance_cm=50.0, db_path=tmp_db)
    cal = db.get_calibration(db_path=tmp_db)
    assert cal["owner_iod_px"] == 60.0
    assert cal["ref_distance_cm"] == 50.0

    db.set_calibration(owner_iod_px=65.0, ref_distance_cm=55.0, db_path=tmp_db)
    cal2 = db.get_calibration(db_path=tmp_db)
    assert cal2["owner_iod_px"] == 65.0
