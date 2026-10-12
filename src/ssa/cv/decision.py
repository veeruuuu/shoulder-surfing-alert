"""Peeker rules + state machine (§4.4) and owner-away protection (§4.5).
§8 contract: decision.step(face_obs_list, now) -> (AlertState, Optional[PeekerEvent], events)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ssa.state import AlertState, FaceObs, PeekerEvent


@dataclass
class _TrackPeekerState:
    consecutive_frames: int = 0
    confirmed: bool = False


class DecisionEngine:
    """Stateful decision engine; one instance per running app, held by pipeline.

    Config values are passed in at construction (from config/defaults.yaml,
    §9), not hardcoded, so dashboard-driven runtime changes (§11 settings
    endpoint) take effect by reconstructing this engine with new values.
    """

    def __init__(
        self,
        near_threshold: float,
        yaw_max: float,
        pitch_max: float,
        confirm_frames: int = 5,
        clear_delay_s: float = 2.0,
        away_after_s: float = 5.0,
        away_action: str = "blur",
        return_grace_s: float = 1.0,
        calibrated: bool = False,
    ) -> None:
        self.near_threshold = near_threshold
        self.yaw_max = yaw_max
        self.pitch_max = pitch_max
        self.confirm_frames = confirm_frames
        self.clear_delay_s = clear_delay_s
        self.away_after_s = away_after_s
        self.away_action = away_action
        self.return_grace_s = return_grace_s
        self.calibrated = calibrated

        self._state: AlertState = "SAFE"
        self._pre_pause_state: AlertState = "SAFE"
        self._track_states: dict[int, _TrackPeekerState] = {}

        self._any_peeker_confirmed_at: Optional[float] = None
        self._current_event: Optional[PeekerEvent] = None

        self._owner_last_seen_at: Optional[float] = None
        self._owner_seen_since: Optional[float] = None

    def update_thresholds(
        self,
        near_threshold: float,
        yaw_max: float,
        pitch_max: float,
        confirm_frames: int,
        clear_delay_s: float,
        away_after_s: float,
        away_action: str,
        return_grace_s: float,
    ) -> None:
        """Apply new config values without resetting in-progress state (current
        AlertState, active PeekerEvent, per-track debounce counters). Used for
        dashboard-driven hot-reload (§11 settings endpoint, §9)."""
        self.near_threshold = near_threshold
        self.yaw_max = yaw_max
        self.pitch_max = pitch_max
        self.confirm_frames = confirm_frames
        self.clear_delay_s = clear_delay_s
        self.away_after_s = away_after_s
        self.away_action = away_action
        self.return_grace_s = return_grace_s

    def pause(self) -> list[str]:
        if self._state != "PAUSED":
            self._pre_pause_state = self._state
            self._state = "PAUSED"
            return ["PAUSE"]
        return []

    def resume(self) -> list[str]:
        if self._state == "PAUSED":
            self._state = self._pre_pause_state
            return ["RESUME"]
        return []

    def step(
        self, face_obs_list: list[FaceObs], now: float
    ) -> tuple[AlertState, Optional[PeekerEvent], list[str]]:
        """Run one frame's worth of decision logic. Mutates each FaceObs's
        `near`, `looking`, `is_peeker` in place (see module docstring)."""
        events: list[str] = []

        if self._state == "PAUSED":
            for face in face_obs_list:
                face.near = False
                face.looking = False
                face.is_peeker = False
            return self._state, None, events

        active_track_ids = set()
        any_peeker_this_frame = False
        owner_seen_this_frame = False

        for face in face_obs_list:
            active_track_ids.add(face.track_id)

            face.near = face.ratio >= self.near_threshold
            face.looking = abs(face.yaw) <= self.yaw_max and abs(face.pitch) <= self.pitch_max

            if face.identity == "owner":
                owner_seen_this_frame = True

            is_candidate = face.identity == "unknown" and face.near and face.looking

            track_state = self._track_states.setdefault(face.track_id, _TrackPeekerState())
            if is_candidate:
                track_state.consecutive_frames += 1
            else:
                track_state.consecutive_frames = 0
                track_state.confirmed = False

            if track_state.consecutive_frames >= self.confirm_frames:
                track_state.confirmed = True

            face.is_peeker = track_state.confirmed
            if face.is_peeker:
                any_peeker_this_frame = True
                if self._current_event is not None:
                    self._current_event.max_ratio = max(self._current_event.max_ratio, face.ratio)
                    self._current_event.min_distance_cm = min(
                        self._current_event.min_distance_cm, face.est_distance_cm
                    )
                    self._current_event.yaw = face.yaw
                    self._current_event.pitch = face.pitch

        for track_id in list(self._track_states.keys()):
            if track_id not in active_track_ids:
                del self._track_states[track_id]

        completed_event: Optional[PeekerEvent] = None

        # --- Owner presence tracking (§4.5) ---
        if owner_seen_this_frame:
            self._owner_last_seen_at = now
            if self._owner_seen_since is None:
                self._owner_seen_since = now
        else:
            self._owner_seen_since = None

        owner_present = (
            self._owner_last_seen_at is not None
            and (now - self._owner_last_seen_at) <= self.away_after_s
        )

        # --- ALERT state transitions (§4.4) ---
        if any_peeker_this_frame:
            self._any_peeker_confirmed_at = now
            if self._state not in ("ALERT", "AWAY"):
                self._state = "ALERT"
                events.append("ALERT_ON")
                self._current_event = PeekerEvent(
                    started_at=now, ended_at=None, max_ratio=0.0,
                    min_distance_cm=float("inf"), yaw=0.0, pitch=0.0, snapshot_path="",
                )
                # Bug fix (found 2026-10-12): this event is created AFTER the
                # face loop above already ran, so the confirming frame's own
                # stats were never recorded -- if the episode ended before a
                # second frame processed, the event kept its placeholder
                # values (max_ratio=0.0, min_distance_cm=inf), and
                # min_distance_cm=Infinity is not valid JSON, silently
                # breaking the dashboard's fetch().json() for the WHOLE
                # intruders list. Populate immediately from this frame's
                # peeker face(s) so the event is never left at placeholders.
                for face in face_obs_list:
                    if face.is_peeker:
                        self._current_event.max_ratio = max(self._current_event.max_ratio, face.ratio)
                        self._current_event.min_distance_cm = min(
                            self._current_event.min_distance_cm, face.est_distance_cm
                        )
                        self._current_event.yaw = face.yaw
                        self._current_event.pitch = face.pitch
        elif self._state == "ALERT":
            if (
                self._any_peeker_confirmed_at is not None
                and (now - self._any_peeker_confirmed_at) >= self.clear_delay_s
            ):
                self._state = "SAFE"
                events.append("ALERT_OFF")
                if self._current_event is not None:
                    self._current_event.ended_at = now
                    completed_event = self._current_event
                    self._current_event = None

        # --- AWAY state transitions (§4.5), only if calibrated ---
        if self.calibrated:
            if self._state in ("SAFE", "ALERT") and not owner_present:
                self._state = "AWAY"
                events.append("AWAY_ON")
                if self.away_action == "lock":
                    events.append("LOCK")
            elif self._state == "AWAY":
                if (
                    owner_seen_this_frame
                    and self._owner_seen_since is not None
                    and (now - self._owner_seen_since) >= self.return_grace_s
                ):
                    self._state = "SAFE"
                    events.append("AWAY_OFF")

        return self._state, completed_event, events
