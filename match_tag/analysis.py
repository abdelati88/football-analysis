"""HTTP API for the automatic analysis: upload a video, watch it run, import
the events it found into the tagger.
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, send_file
from werkzeug.utils import secure_filename

from api import event_from_payload
from jobs import new_job_id
from models import AnalysisJob, Event, Match, db

analysis = Blueprint("analysis", __name__, url_prefix="/api/analysis")

ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def _uploads_dir() -> Path:
    path = Path(current_app.config["UPLOAD_DIR"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_name(filename: str) -> str:
    stem = secure_filename(Path(filename).stem)[:80] or "video"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        suffix = ".mp4"
    return f"{stem}{suffix}"


@analysis.get("/status")
def engine_status():
    """Whether the machine can run an analysis at all, and on what."""
    try:
        from football_ai import weights
    except Exception as exc:
        return jsonify({
            "available": False,
            "reason": "The vision engine is not installed on this server.",
            "detail": str(exc),
            "weights": {},
        })

    # Importing the package is not the same as being able to run it.
    # `football_ai` imports fine with no torch installed — it defers that to
    # first use — so the guard above passes and `pick_device()` then raises
    # ImportError inside the view. On a deployment built without the CV
    # requirements that turned this endpoint into a 500, and the interface
    # calls it on load, so the first thing a visitor met was a server error
    # rather than "analysis is unavailable here".
    try:
        available = weights.available()
        device = weights.pick_device()
    except Exception as exc:
        return jsonify({
            "available": False,
            "reason": "The vision engine is not installed on this server.",
            "detail": str(exc),
            "weights": {},
        })

    missing = [name for name in ("players", "pitch") if not available.get(name)]
    return jsonify({
        "available": not missing,
        "reason": (
            None if not missing
            else f"Missing trained weights: {', '.join(missing)}. "
                 f"Train them with: python training/scripts/train.py {missing[0]}"
        ),
        "weights": available,
        "device": device,
        "busy": current_app.extensions["job_runner"].busy,
        "current_job": current_app.extensions["job_runner"].current_job,
    })


@analysis.post("")
def start_analysis():
    runner = current_app.extensions["job_runner"]
    if runner.busy:
        return jsonify({
            "error": "An analysis is already running. Wait for it to finish, or cancel it.",
            "current_job": runner.current_job,
        }), 409

    upload = request.files.get("video")
    if upload is None or not upload.filename:
        return jsonify({"error": "Attach a video file to analyse."}), 400
    if Path(upload.filename).suffix.lower() not in ALLOWED_SUFFIXES:
        return jsonify({
            "error": f"Unsupported file type. Use one of: {', '.join(sorted(ALLOWED_SUFFIXES))}"
        }), 400

    job_id = new_job_id()
    target_dir = _uploads_dir() / job_id
    target_dir.mkdir(parents=True, exist_ok=True)
    video_path = target_dir / _safe_name(upload.filename)
    upload.save(video_path)

    if video_path.stat().st_size == 0:
        shutil.rmtree(target_dir, ignore_errors=True)
        return jsonify({"error": "That file is empty."}), 400

    def _float_or_none(key: str) -> float | None:
        raw = request.form.get(key)
        try:
            value = float(raw) if raw not in (None, "") else None
        except ValueError:
            return None
        return value if value and value > 0 else None

    options = {
        "team1": request.form.get("team1") or "Team 1",
        "team2": request.form.get("team2") or "Team 2",
        "half": request.form.get("half") or 1,
        "render_video": request.form.get("render_video", "true").lower() != "false",
        "max_seconds": _float_or_none("max_seconds"),
        "sample_fps": _float_or_none("sample_fps"),
    }

    job = AnalysisJob(
        job_id=job_id,
        status="queued",
        stage="queued",
        progress=0.0,
        message="Waiting to start",
        video_name=upload.filename[:300],
        video_path=str(video_path),
        team1=options["team1"],
        team2=options["team2"],
        half=int(options["half"]),
    )
    db.session.add(job)
    db.session.commit()

    if not runner.submit(job_id, video_path, options):
        job.status = "failed"
        job.message = "Another analysis started first. Try again in a moment."
        db.session.commit()
        return jsonify({"error": job.message}), 409

    return jsonify(job.as_dict()), 202


@analysis.get("/<job_id>")
def job_status(job_id: str):
    job = db.session.get(AnalysisJob, job_id)
    if job is None:
        return jsonify({"error": "no such job"}), 404
    want_result = request.args.get("include_result") == "1" and job.status == "done"
    return jsonify(job.as_dict(include_result=want_result))


@analysis.get("")
def list_jobs():
    rows = AnalysisJob.query.order_by(AnalysisJob.created.desc()).limit(25).all()
    return jsonify([j.as_dict() for j in rows])


@analysis.post("/<job_id>/cancel")
def cancel_job(job_id: str):
    job = db.session.get(AnalysisJob, job_id)
    if job is None:
        return jsonify({"error": "no such job"}), 404
    if job.status not in ("queued", "running"):
        return jsonify({"error": f"This analysis has already {job.status}."}), 400
    current_app.extensions["job_runner"].cancel(job_id)
    return jsonify({"success": True})


@analysis.get("/<job_id>/video")
def job_video(job_id: str):
    job = db.session.get(AnalysisJob, job_id)
    if job is None or not job.output_video:
        return jsonify({"error": "no annotated video for this job"}), 404
    path = Path(job.output_video)
    if not path.exists():
        return jsonify({"error": "the annotated video is no longer on disk"}), 410
    return send_file(path, mimetype="video/mp4", conditional=True)


@analysis.post("/<job_id>/save")
def save_as_match(job_id: str):
    """Persist an analysis result as a normal saved match."""
    job = db.session.get(AnalysisJob, job_id)
    if job is None:
        return jsonify({"error": "no such job"}), 404
    if job.status != "done" or not job.result:
        return jsonify({"error": "This analysis has not finished."}), 400

    payload = json.loads(job.result)
    body = request.get_json(silent=True) or {}
    # The browser may send back events the analyst has corrected; fall back to
    # what the pipeline produced if it does not.
    events = body.get("events") or payload.get("events") or []

    match = Match(
        match_name=(body.get("matchName") or job.video_name or "Automatic analysis")[:200],
        team1=(body.get("team1") or job.team1 or "Team 1")[:200],
        team2=(body.get("team2") or job.team2 or "Team 2")[:200],
        video_name=job.video_name,
        source="ai",
    )
    db.session.add(match)
    db.session.flush()
    db.session.bulk_save_objects(
        [event_from_payload(match.match_id, e) for e in events if isinstance(e, dict)]
    )
    job.match_id = match.match_id
    db.session.commit()
    return jsonify({"success": True, "match_id": match.match_id, "events": len(events)}), 201


@analysis.delete("/<job_id>")
def delete_job(job_id: str):
    job = db.session.get(AnalysisJob, job_id)
    if job is None:
        return jsonify({"error": "no such job"}), 404
    if job.status == "running":
        return jsonify({"error": "Cancel the analysis before deleting it."}), 400
    if job.video_path:
        shutil.rmtree(Path(job.video_path).parent, ignore_errors=True)
    db.session.delete(job)
    db.session.commit()
    return jsonify({"success": True})
