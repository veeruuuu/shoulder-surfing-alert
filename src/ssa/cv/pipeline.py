"""Per-frame pipeline wiring: detector -> tracker -> pose -> distance ->
recognizer -> decision (§5.3). Produces FrameResult for AppState."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ssa.cv.decision import DecisionEngine
from ssa.cv.detector import Detector
from ssa.cv.distance import compute as compute_distance
from ssa.cv.pose import PoseSmoother, estimate as estimate_pose
from ssa.cv.recognizer import GalleryEntry, Recognizer
from ssa.cv.tracker import Tracker
from ssa.state import FaceObs, FrameResult, PeekerEvent


class Pipeline:
    """Owns per-track caches (pose smoothing §4.2, embedding cache §4.3) and
    wires the per-frame flow described in §5.3."""

    def __init__(self, config: dict, decision_engine: DecisionEngine,
                 owner_iod_px: Optional[float] = None) -> None:
        self._config = config
        self._detector = Detector(
            max_faces=config["detection"]["max_faces"],
            min_confidence=config["detection"]["min_confidence"],
        )
        self._tracker = Tracker()
        self._recognizer = Recognizer()
        self._pose_alpha = config["gaze"]["smoothing_alpha"]
        self._pose_smoother = PoseSmoother(alpha=self._pose_alpha)
        self._decision_engine = decision_engine
        self._owner_iod_px = owner_iod_px
        self._ref_cm = config["distance"]["owner_ref_distance_cm"]

        self._detect_every_n = config["detection"]["detect_every_n"]
        self._frame_count = 0
        self._last_raw_faces: list = []

        self._embedding_cache: dict[int, tuple[np.ndarray, float]] = {}
        self._reid_interval_s = config["recognition"]["reid_interval_s"]
        self._match_threshold = config["recognition"]["match_threshold"]

        self._gallery: list[GalleryEntry] = []

    def set_gallery(self, gallery: list[GalleryEntry]) -> None:
        self._gallery = gallery

    def set_owner_iod_px(self, owner_iod_px: float) -> None:
        self._owner_iod_px = owner_iod_px

    def update_config(self, config: dict) -> None:
        """Apply new config values without rebuilding the pipeline (keeps
        detector/tracker/recognizer and all per-track caches intact). Camera
        resolution and model paths are NOT updated here -- those still need
        a restart. Used for dashboard-driven hot-reload (§11 settings endpoint).

        The PoseSmoother is only recreated if smoothing_alpha actually
        changed -- recreating it unconditionally on every reload would wipe
        every track's smoothed yaw/pitch history each time (reload runs every
        few seconds), causing a one-frame pose glitch that can spuriously
        reset the confirm_frames debounce in decision.py. Confirmed via live
        testing 2026-10-12: this was making ALERT feel inconsistently slow
        even with confirm_frames back at its default."""
        self._config = config
        self._ref_cm = config["distance"]["owner_ref_distance_cm"]
        self._detect_every_n = config["detection"]["detect_every_n"]
        self._reid_interval_s = config["recognition"]["reid_interval_s"]
        self._match_threshold = config["recognition"]["match_threshold"]

        new_alpha = config["gaze"]["smoothing_alpha"]
        if new_alpha != self._pose_alpha:
            self._pose_alpha = new_alpha
            self._pose_smoother = PoseSmoother(alpha=new_alpha)

    def embed_face(self, frame_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Expose recognizer.embed for calibration/enrollment flows that run
        outside the per-frame loop (e.g. /api/calibrate, /api/trusted/enroll).
        Reuses the already-loaded SFace model rather than creating a second one."""
        return self._recognizer.embed(frame_bgr, landmarks)

    def process_frame(
        self, frame_bgr: np.ndarray
    ) -> tuple[FrameResult, list[str], Optional[PeekerEvent]]:
        now = time.time()
        h, w = frame_bgr.shape[:2]

        if self._frame_count % self._detect_every_n == 0:
            self._last_raw_faces = self._detector.detect(frame_bgr)
        self._frame_count += 1

        tracked = self._tracker.update(self._last_raw_faces)

        face_obs_list: list[FaceObs] = []
        active_ids = set()
        for t in tracked:
            active_ids.add(t.track_id)
            yaw_raw, pitch_raw, roll = estimate_pose(t.landmarks, (h, w))
            yaw, pitch, _ = self._pose_smoother.smooth(t.track_id, yaw_raw, pitch_raw, roll)

            iod_px, ratio, est_cm = compute_distance(
                t.landmarks, (h, w), self._owner_iod_px, self._ref_cm
            )

            identity, trusted_name = self._resolve_identity(t.track_id, frame_bgr, t.landmarks, now)

            face_obs_list.append(FaceObs(
                track_id=t.track_id, bbox=t.bbox, landmarks=t.landmarks,
                iod_px=iod_px, ratio=ratio, est_distance_cm=est_cm,
                yaw=yaw, pitch=pitch, roll=roll,
                identity=identity, trusted_name=trusted_name,
                near=False, looking=False, is_peeker=False,
            ))

        for track_id in list(self._embedding_cache.keys()):
            if track_id not in active_ids:
                del self._embedding_cache[track_id]
                self._pose_smoother.forget(track_id)

        state, completed_event, events = self._decision_engine.step(face_obs_list, now)
        owner_present = any(f.identity == "owner" for f in face_obs_list)

        elapsed = max(1e-6, time.time() - now)
        fps = 1.0 / elapsed
        result = FrameResult(
            timestamp=now, faces=face_obs_list, state=state, fps=fps,
            owner_present=owner_present,
        )
        return result, events, completed_event

    def _resolve_identity(self, track_id: int, frame_bgr: np.ndarray, landmarks: np.ndarray, now: float):
        """§4.3: embed only new tracks, re-check every reid_interval_s; cache per track."""
        if not self._gallery:
            return "unknown", None

        cached = self._embedding_cache.get(track_id)
        if cached is None or (now - cached[1]) >= self._reid_interval_s:
            try:
                emb = self._recognizer.embed(frame_bgr, landmarks)
            except Exception:
                return "unknown", None
            self._embedding_cache[track_id] = (emb, now)
        else:
            emb = cached[0]

        return self._recognizer.match(emb, self._gallery, threshold=self._match_threshold)

    def close(self) -> None:
        self._detector.close()
