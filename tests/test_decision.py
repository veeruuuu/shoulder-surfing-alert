"""Â§14: decision.py unit tests â€” state transitions, debounce, clear delay,
one event per episode."""
from __future__ import annotations

import numpy as np

from ssa.cv.decision import DecisionEngine
from ssa.state import FaceObs


def _face(track_id, identity="unknown", ratio=0.5, yaw=0.0, pitch=0.0, distance_cm=40.0):
    return FaceObs(
        track_id=track_id,
        bbox=(0, 0, 10, 10),
        landmarks=np.zeros((478, 3), dtype=np.float32),
        iod_px=60.0,
        ratio=ratio,
        est_distance_cm=distance_cm,
        yaw=yaw,
        pitch=pitch,
        roll=0.0,
        identity=identity,
        trusted_name=None,
        near=False,
        looking=False,
        is_peeker=False,
    )


def _engine(**overrides):
    defaults = dict(
        near_threshold=0.4, yaw_max=40, pitch_max=30,
        confirm_frames=3, clear_delay_s=1.0,
        away_after_s=2.0, away_action="blur", return_grace_s=0.5,
        calibrated=False,
    )
    defaults.update(overrides)
    return DecisionEngine(**defaults)


def test_debounce_requires_consecutive_frames():
    engine = _engine()
    t = 0.0
    for i in range(2):  # fewer than confirm_frames=3
        state, event, events = engine.step([_face(1)], now=t)
        t += 0.1
    assert state == "SAFE"
    assert events == []

    state, event, events = engine.step([_face(1)], now=t)  # 3rd consecutive
    assert state == "ALERT"
    assert "ALERT_ON" in events


def test_non_consecutive_candidate_frames_do_not_confirm():
    engine = _engine()
    t = 0.0
    engine.step([_face(1)], now=t); t += 0.1
    engine.step([_face(1, ratio=0.1)], now=t); t += 0.1  # breaks the streak (not near)
    state, _, events = engine.step([_face(1)], now=t)
    assert state == "SAFE"
    assert events == []


def test_owner_or_trusted_never_confirms_as_peeker():
    engine = _engine()
    t = 0.0
    for identity in ("owner", "trusted"):
        engine = _engine()
        for _ in range(5):
            state, _, events = engine.step([_face(1, identity=identity)], now=t)
            t += 0.1
        assert state == "SAFE"
        assert events == []


def test_clear_delay_then_one_completed_event_per_episode():
    engine = _engine()
    t = 0.0
    for _ in range(3):
        state, event, events = engine.step([_face(1)], now=t)
        t += 0.1
    assert state == "ALERT" and event is None

    # Peeker leaves; state stays ALERT until clear_delay_s elapses.
    state, event, events = engine.step([], now=t)
    assert state == "ALERT"
    assert event is None
    assert events == []

    t += 1.0  # >= clear_delay_s
    state, event, events = engine.step([], now=t)
    assert state == "SAFE"
    assert "ALERT_OFF" in events
    assert event is not None
    assert event.started_at == 0.2
    assert event.ended_at == t


def test_two_peekers_log_one_event_per_episode():
    engine = _engine()
    t = 0.0
    for _ in range(3):
        state, event, events = engine.step([_face(1), _face(2)], now=t)
        t += 0.1
    assert state == "ALERT"

    t += 1.0
    state, event, events = engine.step([], now=t)
    assert state == "SAFE"
    assert event is not None  # exactly one PeekerEvent for the whole episode


def test_owner_away_after_timeout_and_return_clears_it():
    engine = _engine(calibrated=True, away_after_s=2.0, return_grace_s=0.5)
    t = 0.0
    state, _, events = engine.step([_face(1, identity="owner")], now=t)
    assert state == "SAFE"

    t += 2.5  # owner not seen for > away_after_s
    state, _, events = engine.step([], now=t)
    assert state == "AWAY"
    assert "AWAY_ON" in events

    # A stranger/trusted face does not clear AWAY.
    state, _, events = engine.step([_face(1, identity="trusted")], now=t)
    assert state == "AWAY"

    # Owner returns, held for return_grace_s.
    t += 0.1
    state, _, events = engine.step([_face(1, identity="owner")], now=t)
    assert state == "AWAY"  # not yet held long enough
    t += 0.6
    state, _, events = engine.step([_face(1, identity="owner")], now=t)
    assert state == "SAFE"
    assert "AWAY_OFF" in events


def test_pause_and_resume():
    engine = _engine()
    events = engine.pause()
    assert events == ["PAUSE"]
    state, event, events = engine.step([_face(1)], now=0.0)
    assert state == "PAUSED"
    events = engine.resume()
    assert events == ["RESUME"]


def test_single_frame_episode_records_real_stats_not_placeholders():
    """Regression test (2026-10-12): an episode confirmed and cleared within
    one step() call must not leave the event at its placeholder values
    (max_ratio=0.0, min_distance_cm=inf), since inf is invalid JSON and
    broke the dashboard's intruder log for the whole list."""
    engine = _engine(confirm_frames=1, clear_delay_s=0.0)
    t = 0.0
    state, event, events = engine.step(
        [_face(1, ratio=0.9, distance_cm=42.0, yaw=5.0, pitch=3.0)], now=t
    )
    assert state == "ALERT"
    t += 0.1
    state, event, events = engine.step([], now=t)
    assert state == "SAFE"
    assert event is not None
    assert event.max_ratio == 0.9
    assert event.min_distance_cm == 42.0
    assert event.min_distance_cm != float("inf")


def test_single_frame_episode_records_real_stats_not_placeholders():
    """Regression test (2026-10-12): an episode confirmed and cleared within
    one step() call must not leave the event at its placeholder values
    (max_ratio=0.0, min_distance_cm=inf), since inf is invalid JSON and
    broke the dashboard's intruder log for the whole list."""
    engine = _engine(confirm_frames=1, clear_delay_s=0.0)
    t = 0.0
    state, event, events = engine.step(
        [_face(1, ratio=0.9, distance_cm=42.0, yaw=5.0, pitch=3.0)], now=t
    )
    assert state == "ALERT"
    t += 0.1
    state, event, events = engine.step([], now=t)
    assert state == "SAFE"
    assert event is not None
    assert event.max_ratio == 0.9
    assert event.min_distance_cm == 42.0
    assert event.min_distance_cm != float("inf")
