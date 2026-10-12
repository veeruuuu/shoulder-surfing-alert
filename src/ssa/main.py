"""Starts threads, owns Tk main loop (§5.1, B6). Run: python -m ssa.main"""
from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from typing import Any, Optional

import cv2
import numpy as np

from ssa.config import load_config
from ssa.cv.capture import Capture, CameraAccessError
from ssa.cv.decision import DecisionEngine
from ssa.cv.pipeline import Pipeline
from ssa.cv.recognizer import GalleryEntry
from ssa.dashboard.app import create_app
from ssa.overlay.overlay import Overlay
from ssa.platform_utils import lock_screen, set_dpi_aware
from ssa.state import AppState, FaceObs, FrameResult
from ssa.storage import db, snapshots


def _build_decision_engine(config: dict, calibrated: bool) -> DecisionEngine:
    preset = config["distance"]["preset"]
    near_threshold = config["distance"]["presets"][preset]
    return DecisionEngine(
        near_threshold=near_threshold,
        yaw_max=config["gaze"]["yaw_max"],
        pitch_max=config["gaze"]["pitch_max"],
        confirm_frames=config["decision"]["confirm_frames"],
        clear_delay_s=config["decision"]["clear_delay_s"],
        away_after_s=config["decision"]["away_after_s"],
        away_action=config["decision"]["away_action"],
        return_grace_s=config["decision"]["return_grace_s"],
        calibrated=calibrated,
    )


def _draw_preview(frame: np.ndarray, result: FrameResult) -> bytes:
    annotated = frame.copy()
    for face in result.faces:
        x, y, w, h = face.bbox
        color = (0, 0, 255) if face.is_peeker else (0, 255, 0)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        label = f"{face.identity} r={face.ratio:.2f} yaw={face.yaw:.0f} pitch={face.pitch:.0f}"
        cv2.putText(
            annotated, label, (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        )
    cv2.putText(
        annotated, f"state={result.state} fps={result.fps:.1f}", (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )
    ok, buf = cv2.imencode(".jpg", annotated)
    return buf.tobytes() if ok else b""


def _prepare_snapshot_frame(
    frame: np.ndarray, faces: list[FaceObs], blur_other_faces: bool
) -> np.ndarray:
    if not blur_other_faces:
        return frame

    snapshot = frame.copy()
    for face in faces:
        if face.is_peeker:
            continue
        x, y, w, h = face.bbox
        x, y = max(0, x), max(0, y)
        w = min(w, snapshot.shape[1] - x)
        h = min(h, snapshot.shape[0] - y)
        if w <= 0 or h <= 0:
            continue
        roi = snapshot[y:y + h, x:x + w]
        blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
        snapshot[y:y + h, x:x + w] = blurred_roi
    return snapshot


def _handle_cv_task(
    req: dict[str, Any], frame: np.ndarray, result: FrameResult,
    pipeline: Pipeline, decision_engine: DecisionEngine, app_state: AppState,
) -> None:
    kind = req.get("kind")
    faces = result.faces

    if len(faces) != 1:
        app_state.set_cv_task_result({
            "success": False,
            "error": f"Expected exactly 1 face in frame, found {len(faces)}. "
                     "Make sure only you are visible to the camera.",
        })
        return

    face = faces[0]

    if kind == "calibrate":
        ref_cm = req["ref_distance_cm"]
        try:
            emb = pipeline.embed_face(frame, face.landmarks)
        except Exception as exc:  # noqa: BLE001
            app_state.set_cv_task_result({
                "success": False, "error": f"Could not compute face embedding: {exc}",
            })
            return

        db.set_calibration(owner_iod_px=face.iod_px, ref_distance_cm=ref_cm)
        for existing in db.list_trusted_faces():
            if existing["is_owner"]:
                db.delete_trusted_face(existing["id"])
        db.add_trusted_face(name="Owner", embedding=emb, is_owner=True)

        pipeline.set_owner_iod_px(face.iod_px)
        gallery = [
            GalleryEntry(name=t["name"], is_owner=t["is_owner"], embedding=t["embedding"])
            for t in db.list_trusted_faces()
        ]
        pipeline.set_gallery(gallery)

        decision_engine.calibrated = True
        app_state.set_calibrated(True)
        app_state.set_cv_task_result({"success": True, "error": None})

    elif kind == "enroll":
        name = req["name"]
        try:
            emb = pipeline.embed_face(frame, face.landmarks)
        except Exception as exc:  # noqa: BLE001
            app_state.set_cv_task_result({
                "success": False, "error": f"Could not compute face embedding: {exc}",
            })
            return

        db.add_trusted_face(name=name, embedding=emb, is_owner=False)
        gallery = [
            GalleryEntry(name=t["name"], is_owner=t["is_owner"], embedding=t["embedding"])
            for t in db.list_trusted_faces()
        ]
        pipeline.set_gallery(gallery)
        app_state.set_cv_task_result({"success": True, "error": None})

    else:
        app_state.set_cv_task_result({"success": False, "error": f"Unknown task kind: {kind}"})


def _run_cv_loop(config: dict, app_state: AppState, event_queue: "queue.Queue[str]",
                  decision_engine: DecisionEngine, cap: Capture) -> None:
    cal = db.get_calibration()
    owner_iod_px = cal["owner_iod_px"] if cal else None

    pipeline = Pipeline(config, decision_engine, owner_iod_px=owner_iod_px)

    trusted = db.list_trusted_faces()
    gallery = [
        GalleryEntry(name=t["name"], is_owner=t["is_owner"], embedding=t["embedding"])
        for t in trusted
    ]
    pipeline.set_gallery(gallery)

    fps_ema: Optional[float] = None
    fps_alpha = 0.1
    last_frame_time = time.time()

    retention_days = config["storage"]["snapshot_retention_days"]
    cleanup_interval_s = 24 * 3600
    last_cleanup_at = 0.0

    reload_interval_s = 5.0
    last_reload_at = time.time()

    pending_snapshot_frame: Optional[np.ndarray] = None
    pending_snapshot_faces: list[FaceObs] = []

    try:
        app_state.set_camera_active(True)
        app_state.set_camera_error(None)
        while not app_state.is_shutting_down():
            if app_state.is_paused():
                for ev in decision_engine.pause():
                    event_queue.put(ev)
            else:
                for ev in decision_engine.resume():
                    event_queue.put(ev)

            now_cleanup = time.time()
            if now_cleanup - last_cleanup_at >= cleanup_interval_s:
                deleted = snapshots.cleanup_old_snapshots(retention_days)
                if deleted:
                    print(f"Snapshot retention cleanup: deleted {deleted} file(s) older than {retention_days} days.")
                last_cleanup_at = now_cleanup

            now_reload = time.time()
            if now_reload - last_reload_at >= reload_interval_s:
                config = load_config()
                decision_engine.update_thresholds(
                    near_threshold=config["distance"]["presets"][config["distance"]["preset"]],
                    yaw_max=config["gaze"]["yaw_max"],
                    pitch_max=config["gaze"]["pitch_max"],
                    confirm_frames=config["decision"]["confirm_frames"],
                    clear_delay_s=config["decision"]["clear_delay_s"],
                    away_after_s=config["decision"]["away_after_s"],
                    away_action=config["decision"]["away_action"],
                    return_grace_s=config["decision"]["return_grace_s"],
                )
                pipeline.update_config(config)
                retention_days = config["storage"]["snapshot_retention_days"]
                last_reload_at = now_reload

            frame = cap.read()
            if frame is None:
                time.sleep(0.05)
                continue

            now = time.time()
            dt = now - last_frame_time
            last_frame_time = now
            if dt > 0:
                instant_fps = 1.0 / dt
                fps_ema = instant_fps if fps_ema is None else (
                    fps_alpha * instant_fps + (1 - fps_alpha) * fps_ema
                )

            result, events, completed_event = pipeline.process_frame(frame)
            if fps_ema is not None:
                result.fps = round(fps_ema, 1)
            app_state.set_frame_result(result)
            app_state.set_preview_jpeg(_draw_preview(frame, result))

            if "ALERT_ON" in events:
                pending_snapshot_frame = frame.copy()
                pending_snapshot_faces = result.faces

            task_req = app_state.pop_cv_task_request()
            if task_req is not None:
                _handle_cv_task(task_req, frame, result, pipeline, decision_engine, app_state)

            for ev in events:
                event_queue.put(ev)

            if completed_event is not None:
                snapshot_source = pending_snapshot_frame if pending_snapshot_frame is not None else frame
                snapshot_faces = pending_snapshot_faces
                blur_other_faces = config["storage"]["snapshot_blur_other_faces"]
                snapshot_frame = _prepare_snapshot_frame(snapshot_source, snapshot_faces, blur_other_faces)

                snapshot_path = snapshots.save_snapshot(snapshot_frame, completed_event.started_at)
                db.add_intruder_event(
                    started_at=completed_event.started_at,
                    ended_at=completed_event.ended_at,
                    max_ratio=completed_event.max_ratio,
                    min_distance_cm=completed_event.min_distance_cm,
                    yaw=completed_event.yaw,
                    pitch=completed_event.pitch,
                    snapshot_path=snapshot_path,
                )
                pending_snapshot_frame = None
                pending_snapshot_faces = []
    finally:
        app_state.set_camera_active(False)
        pipeline.close()


def _cv_thread_main(
    config: dict,
    app_state: AppState,
    event_queue: "queue.Queue[str]",
    decision_engine: DecisionEngine,
) -> None:
    """Outer loop (§15 B7: permission checks): if the camera can't be opened
    (permission denied, in use by another app, not found), store a clear
    error in AppState for the dashboard to show, and keep retrying every few
    seconds instead of crashing the whole app -- recovers automatically if
    the user grants permission or frees up the camera, no restart needed."""
    retry_interval_s = 3.0
    while not app_state.is_shutting_down():
        try:
            with Capture(
                index=config["camera"]["index"],
                width=config["camera"]["width"],
                height=config["camera"]["height"],
            ) as cap:
                _run_cv_loop(config, app_state, event_queue, decision_engine, cap)
            return  # loop exited cleanly (shutdown requested)
        except CameraAccessError as exc:
            app_state.set_camera_active(False)
            app_state.set_camera_error(str(exc))
            print(f"Camera error: {exc}")
            print(f"Retrying in {retry_interval_s:.0f}s...")
            for _ in range(int(retry_interval_s * 10)):
                if app_state.is_shutting_down():
                    return
                time.sleep(0.1)
        except Exception as exc:  # noqa: BLE001 - unexpected error, surface and stop
            app_state.set_camera_active(False)
            app_state.set_camera_error(f"Unexpected error: {exc}")
            print(f"CV thread error: {exc}")
            return


def _web_thread_main(config: dict, app_state: AppState) -> None:
    app = create_app(app_state, config)
    app.run(
        host=config["dashboard"]["host"],
        port=config["dashboard"]["port"],
        threaded=True,
        use_reloader=False,
    )


def main() -> None:
    config = load_config()
    db.init_db()

    set_dpi_aware()
    root = tk.Tk()
    root.withdraw()  # hidden root (§5.1)

    overlay = Overlay(
        root,
        blur_kernel=config["overlay"]["blur_kernel"],
        downscale=config["overlay"]["downscale"],
    )
    banner_peeker = config["overlay"]["banner_text"]
    banner_away = "Screen hidden: owner away"

    app_state = AppState()
    event_queue: "queue.Queue[str]" = queue.Queue()

    cal = db.get_calibration()
    calibrated = cal is not None
    app_state.set_calibrated(calibrated)
    decision_engine = _build_decision_engine(config, calibrated)

    cv_thread = threading.Thread(
        target=_cv_thread_main,
        args=(config, app_state, event_queue, decision_engine),
        daemon=True,
    )
    cv_thread.start()

    web_thread = threading.Thread(
        target=_web_thread_main, args=(config, app_state), daemon=True,
    )
    web_thread.start()

    def poll_queue() -> None:
        try:
            while True:
                ev = event_queue.get_nowait()
                if ev == "ALERT_ON":
                    overlay.show(banner_peeker)
                elif ev == "AWAY_ON":
                    overlay.show(banner_away)
                elif ev in ("ALERT_OFF", "AWAY_OFF"):
                    overlay.hide()
                elif ev == "LOCK":
                    lock_screen()
        except queue.Empty:
            pass
        root.after(100, poll_queue)

    root.after(100, poll_queue)

    url = f"http://{config['dashboard']['host']}:{config['dashboard']['port']}"
    print("Shoulder Surfing Alert running.")
    print(f"Dashboard: {url}/")
    if not calibrated:
        print("Not calibrated yet: calibrate from the dashboard or POST /api/calibrate.")
    print("Press Ctrl+C in this terminal to quit.")

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        app_state.request_shutdown()
        cv_thread.join(timeout=3)


if __name__ == "__main__":
    main()
