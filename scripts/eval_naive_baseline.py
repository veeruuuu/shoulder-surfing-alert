"""§14 baseline comparison: naive "2 faces means blur" vs the real system's
distance+gaze filtering. Standalone script — does not touch data/app.db,
snapshots, or calibration. Prints episode start/end to console; tally results
into docs/eval_results.md yourself.

Usage:
  python scripts/eval_naive_baseline.py --mode naive
  python scripts/eval_naive_baseline.py --mode full

"naive": alerts whenever >=2 faces are in frame for confirm_frames consecutive
frames (no distance/gaze/identity filtering at all).
"full": alerts using the real decision rules (distance+gaze+identity), same as
main.py, but requires calibration to already exist in data/app.db (read-only).
Run the same real scenario for both modes and compare episode counts.
"""
from __future__ import annotations

import argparse
import time

from ssa.config import load_config
from ssa.cv.capture import Capture
from ssa.cv.decision import DecisionEngine
from ssa.cv.detector import Detector
from ssa.cv.distance import compute as compute_distance
from ssa.cv.pose import PoseSmoother, estimate as estimate_pose
from ssa.cv.recognizer import GalleryEntry, Recognizer
from ssa.cv.tracker import Tracker
from ssa.state import FaceObs
from ssa.storage import db


def run_naive(config: dict, duration_s: float) -> int:
    confirm_frames = config["decision"]["confirm_frames"]
    clear_delay_s = config["decision"]["clear_delay_s"]

    consecutive = 0
    in_episode = False
    episode_count = 0
    last_multi_face_at = None

    with Capture(
        index=config["camera"]["index"], width=config["camera"]["width"], height=config["camera"]["height"]
    ) as cap:
        detector = Detector(max_faces=config["detection"]["max_faces"], min_confidence=config["detection"]["min_confidence"])
        start = time.time()
        try:
            while time.time() - start < duration_s:
                frame = cap.read()
                if frame is None:
                    continue
                faces = detector.detect(frame)
                now = time.time()

                if len(faces) >= 2:
                    consecutive += 1
                    last_multi_face_at = now
                else:
                    consecutive = 0

                if not in_episode and consecutive >= confirm_frames:
                    in_episode = True
                    episode_count += 1
                    print(f"[naive] EPISODE START #{episode_count} at t={now - start:.1f}s ({len(faces)} faces)")
                elif in_episode and last_multi_face_at is not None and (now - last_multi_face_at) >= clear_delay_s:
                    in_episode = False
                    print(f"[naive] episode end at t={now - start:.1f}s")
        finally:
            detector.close()

    return episode_count


def run_full(config: dict, duration_s: float) -> int:
    cal = db.get_calibration()
    if cal is None:
        raise SystemExit(
            "No calibration found in data/app.db. Calibrate via the dashboard first "
            "(python -m ssa.main, then http://127.0.0.1:5000/calibration) before running --mode full."
        )
    owner_iod_px = cal["owner_iod_px"]

    preset = config["distance"]["preset"]
    near_threshold = config["distance"]["presets"][preset]
    decision_engine = DecisionEngine(
        near_threshold=near_threshold,
        yaw_max=config["gaze"]["yaw_max"],
        pitch_max=config["gaze"]["pitch_max"],
        confirm_frames=config["decision"]["confirm_frames"],
        clear_delay_s=config["decision"]["clear_delay_s"],
        away_after_s=config["decision"]["away_after_s"],
        away_action=config["decision"]["away_action"],
        return_grace_s=config["decision"]["return_grace_s"],
        calibrated=True,
    )

    trusted = db.list_trusted_faces()
    gallery = [GalleryEntry(name=t["name"], is_owner=t["is_owner"], embedding=t["embedding"]) for t in trusted]

    detector = Detector(max_faces=config["detection"]["max_faces"], min_confidence=config["detection"]["min_confidence"])
    tracker = Tracker()
    recognizer = Recognizer()
    pose_smoother = PoseSmoother(alpha=config["gaze"]["smoothing_alpha"])
    ref_cm = config["distance"]["owner_ref_distance_cm"]

    episode_count = 0
    prev_state = "SAFE"

    with Capture(
        index=config["camera"]["index"], width=config["camera"]["width"], height=config["camera"]["height"]
    ) as cap:
        start = time.time()
        try:
            while time.time() - start < duration_s:
                frame = cap.read()
                if frame is None:
                    continue
                h, w = frame.shape[:2]
                raw_faces = detector.detect(frame)
                tracked = tracker.update(raw_faces)

                face_obs_list = []
                for t in tracked:
                    yaw_raw, pitch_raw, roll = estimate_pose(t.landmarks, (h, w))
                    yaw, pitch, _ = pose_smoother.smooth(t.track_id, yaw_raw, pitch_raw, roll)
                    iod_px, ratio, est_cm = compute_distance(t.landmarks, (h, w), owner_iod_px, ref_cm)
                    try:
                        emb = recognizer.embed(frame, t.landmarks)
                        identity, name = recognizer.match(emb, gallery, threshold=config["recognition"]["match_threshold"])
                    except Exception:
                        identity, name = "unknown", None
                    face_obs_list.append(FaceObs(
                        track_id=t.track_id, bbox=t.bbox, landmarks=t.landmarks,
                        iod_px=iod_px, ratio=ratio, est_distance_cm=est_cm,
                        yaw=yaw, pitch=pitch, roll=roll, identity=identity, trusted_name=name,
                        near=False, looking=False, is_peeker=False,
                    ))

                now = time.time()
                state, completed_event, events = decision_engine.step(face_obs_list, now)
                if state == "ALERT" and prev_state != "ALERT":
                    episode_count += 1
                    print(f"[full] EPISODE START #{episode_count} at t={now - start:.1f}s")
                elif state != "ALERT" and prev_state == "ALERT":
                    print(f"[full] episode end at t={now - start:.1f}s")
                prev_state = state
        finally:
            detector.close()

    return episode_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["naive", "full"], required=True)
    parser.add_argument("--duration", type=float, default=600.0, help="seconds (default 600 = 10 min)")
    args = parser.parse_args()

    config = load_config()
    print(f"Running {args.mode} baseline for {args.duration:.0f}s. Press Ctrl+C to stop early.")
    try:
        if args.mode == "naive":
            count = run_naive(config, args.duration)
        else:
            count = run_full(config, args.duration)
    except KeyboardInterrupt:
        count = -1
        print("\nStopped early.")

    print(f"\n[{args.mode}] total episodes triggered: {count if count >= 0 else '(stopped early, see console log above)'}")
    print("Record this in docs/eval_results.md under 'Baseline Comparison'.")


if __name__ == "__main__":
    main()
