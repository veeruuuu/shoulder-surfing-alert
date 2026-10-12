"""Flask dashboard API (§11, B4) and 5-page UI (B5)."""
from __future__ import annotations

import json
import time

from flask import Flask, Response, jsonify, render_template, request

from ssa.config import load_config, save_settings
from ssa.state import AppState
from ssa.storage import db, snapshots


def _wait_for_cv_task_result(app_state: AppState, timeout_s: float = 10.0) -> dict:
    start = time.time()
    while time.time() - start < timeout_s:
        result = app_state.pop_cv_task_result()
        if result is not None:
            return result
        time.sleep(0.1)
    return {"success": False, "error": "Timed out waiting for the camera thread."}


def _sanitize_event(e: dict) -> dict:
    """Replace non-JSON-safe float values (inf/-inf/nan) with null. Python's
    json.dumps writes bare Infinity/NaN tokens, which are NOT valid JSON and
    silently break JSON.parse() in the browser -- found 2026-10-12. Root cause
    fixed in decision.py (events are now populated from the confirming frame
    immediately); this sanitizer protects against any already-saved rows and
    any future edge case that slips through."""
    d = dict(e)
    for key in ("max_ratio", "min_distance_cm", "yaw", "pitch"):
        val = d.get(key)
        if isinstance(val, float) and (val != val or val in (float("inf"), float("-inf"))):
            d[key] = None
    return d


def create_app(app_state: AppState, config: dict) -> Flask:
    app = Flask(__name__)

    # --- UI pages (B5: 5 pages) ---

    @app.route("/", methods=["GET"])
    def page_live():
        return render_template("live.html", active="live")

    @app.route("/calibration", methods=["GET"])
    def page_calibration():
        return render_template("calibration.html", active="calibration")

    @app.route("/trusted", methods=["GET"])
    def page_trusted():
        return render_template("trusted.html", active="trusted")

    @app.route("/intruders", methods=["GET"])
    def page_intruders():
        return render_template("intruders.html", active="intruders")

    @app.route("/settings", methods=["GET"])
    def page_settings():
        return render_template("settings.html", active="settings")

    # --- Status & stream ---

    @app.route("/api/status", methods=["GET"])
    def status():
        result = app_state.get_frame_result()
        return jsonify({
            "state": result.state if result else "SAFE",
            "fps": round(result.fps, 1) if result else 0.0,
            "face_count": len(result.faces) if result else 0,
            "calibrated": app_state.is_calibrated(),
            "camera_active": app_state.is_camera_active(),
            "paused": app_state.is_paused(),
            "camera_error": app_state.get_camera_error(),
        })

    def _generate_mjpeg():
        boundary = b"--frame"
        while True:
            jpeg = app_state.get_preview_jpeg()
            if jpeg is not None:
                yield (
                    boundary + b"\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            time.sleep(1.0 / 15)

    @app.route("/api/stream", methods=["GET"])
    def stream():
        return Response(
            _generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    # --- Calibration ---

    @app.route("/api/calibrate", methods=["POST"])
    def calibrate():
        body = request.get_json(silent=True) or {}
        ref_cm = body.get("ref_distance_cm", config["distance"]["owner_ref_distance_cm"])
        app_state.request_cv_task("calibrate", ref_distance_cm=ref_cm)
        result = _wait_for_cv_task_result(app_state)
        status_code = 200 if result.get("success") else (504 if "Timed out" in result.get("error", "") else 400)
        return jsonify(result), status_code

    # --- Trusted faces ---

    @app.route("/api/trusted", methods=["GET"])
    def list_trusted():
        faces = db.list_trusted_faces()
        return jsonify([
            {"id": f["id"], "name": f["name"], "is_owner": f["is_owner"], "created_at": f["created_at"]}
            for f in faces
        ])

    @app.route("/api/trusted", methods=["POST"])
    def enroll_trusted():
        body = request.get_json(silent=True) or {}
        name = body.get("name")
        if not name:
            return jsonify({"success": False, "error": "name is required"}), 400
        app_state.request_cv_task("enroll", name=name)
        result = _wait_for_cv_task_result(app_state)
        status_code = 200 if result.get("success") else (504 if "Timed out" in result.get("error", "") else 400)
        return jsonify(result), status_code

    @app.route("/api/trusted/<int:face_id>", methods=["DELETE"])
    def delete_trusted(face_id: int):
        db.delete_trusted_face(face_id)
        return jsonify({"success": True})

    # --- Intruder events ---

    @app.route("/api/intruders", methods=["GET"])
    def list_intruders():
        return jsonify([_sanitize_event(e) for e in db.list_intruder_events()])

    @app.route("/api/intruders/<int:event_id>/snapshot", methods=["GET"])
    def intruder_snapshot(event_id: int):
        events = db.list_intruder_events()
        match = next((e for e in events if e["id"] == event_id), None)
        if match is None or not match["snapshot_path"]:
            return jsonify({"error": "not found"}), 404
        snapshot_file = snapshots.REPO_ROOT / match["snapshot_path"]
        if not snapshot_file.exists():
            return jsonify({"error": "snapshot file missing on disk"}), 404
        return Response(snapshot_file.read_bytes(), mimetype="image/jpeg")

    @app.route("/api/intruders/<int:event_id>", methods=["DELETE"])
    def delete_intruder(event_id: int):
        db.delete_intruder_event(event_id)
        return jsonify({"success": True})

    # --- Settings ---

    @app.route("/api/settings", methods=["GET"])
    def get_settings():
        return jsonify(load_config())

    @app.route("/api/settings", methods=["POST"])
    def update_settings():
        body = request.get_json(silent=True) or {}
        if not isinstance(body, dict):
            return jsonify({"success": False, "error": "body must be a JSON object"}), 400
        save_settings(body)
        return jsonify({"success": True, "config": load_config()})

    # --- Engine control ---

    @app.route("/api/engine/pause", methods=["POST"])
    def engine_pause():
        app_state.set_paused(True)
        return jsonify({"success": True, "paused": True})

    @app.route("/api/engine/resume", methods=["POST"])
    def engine_resume():
        app_state.set_paused(False)
        return jsonify({"success": True, "paused": False})

    # --- Stats ---

    @app.route("/api/stats", methods=["GET"])
    def stats():
        events = db.list_intruder_events()
        completed = [e for e in events if e["duration_s"] is not None]
        total_duration = sum(e["duration_s"] for e in completed)
        avg_duration = total_duration / len(completed) if completed else 0.0
        return jsonify({
            "total_events": len(events),
            "avg_duration_s": round(avg_duration, 2),
            "total_duration_s": round(total_duration, 2),
        })

    # --- Server-Sent Events ---

    @app.route("/api/events", methods=["GET"])
    def sse_events():
        def _generate():
            last_state = None
            while True:
                result = app_state.get_frame_result()
                current_state = result.state if result else "SAFE"
                if current_state != last_state:
                    payload = json.dumps({"state": current_state, "timestamp": time.time()})
                    yield f"data: {payload}\n\n"
                    last_state = current_state
                time.sleep(0.5)

        return Response(_generate(), mimetype="text/event-stream")

    return app


