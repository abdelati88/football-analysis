"""Running the vision pipeline without blocking the web server.

Analysis takes minutes, so a request cannot wait for it. Each job runs on a
worker thread and reports progress into the database, which the browser polls.
Putting progress in the database rather than in memory means a reload — of the
page or of the server — does not lose track of a run in flight.

One job at a time. The models want most of the machine's memory, and two
concurrent runs on a laptop would take longer together than one after the other.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask

from models import AnalysisJob, db

# Progress is written at most this often, so a 90-minute video does not issue
# tens of thousands of database writes.
_MIN_PROGRESS_INTERVAL = 0.7

# Weight of each pipeline stage in the overall bar. The main analysis pass
# dominates; rendering is a cheaper second pass over the same video.
_STAGE_WEIGHTS = {
    "queued": (0.00, 0.00),
    "survey": (0.00, 0.06),
    "analyse": (0.06, 0.72),
    "derive": (0.72, 0.80),
    "render": (0.80, 1.00),
}


class JobRunner:
    """Owns the worker thread and the one-job-at-a-time rule."""

    def __init__(self, app: Flask):
        self.app = app
        self._lock = threading.Lock()
        self._current: str | None = None
        self._cancel = threading.Event()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._current is not None

    @property
    def current_job(self) -> str | None:
        with self._lock:
            return self._current

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            if self._current != job_id:
                return False
        self._cancel.set()
        return True

    def submit(self, job_id: str, video_path: Path, options: dict[str, Any]) -> bool:
        with self._lock:
            if self._current is not None:
                return False
            self._current = job_id
            self._cancel.clear()

        threading.Thread(
            target=self._run,
            args=(job_id, video_path, options),
            daemon=True,
            name=f"analysis-{job_id[:8]}",
        ).start()
        return True

    # ---- the worker ---------------------------------------------------------

    def _update(self, job_id: str, **fields: Any) -> None:
        job = db.session.get(AnalysisJob, job_id)
        if job is None:
            return
        for key, value in fields.items():
            setattr(job, key, value)
        db.session.commit()

    def _run(self, job_id: str, video_path: Path, options: dict[str, Any]) -> None:
        with self.app.app_context():
            try:
                self._update(
                    job_id, status="running", stage="survey", progress=0.0,
                    message="Loading the models",
                )

                # Imported here rather than at module scope: the vision stack
                # pulls in torch, which costs seconds of import time and
                # hundreds of megabytes. A server that never runs an analysis
                # should not pay for it.
                from football_ai.pipeline import PipelineConfig, analyse

                last_write = [0.0]

                def on_progress(stage: str, fraction: float, message: str) -> None:
                    if self._cancel.is_set():
                        raise KeyboardInterrupt("cancelled by the user")
                    now = time.monotonic()
                    if fraction < 0.999 and now - last_write[0] < _MIN_PROGRESS_INTERVAL:
                        return
                    last_write[0] = now
                    start, end = _STAGE_WEIGHTS.get(stage, (0.0, 1.0))
                    self._update(
                        job_id,
                        stage=stage,
                        progress=start + (end - start) * fraction,
                        message=message[:400],
                    )

                config = PipelineConfig(
                    team_names={
                        1: options.get("team1") or "Team 1",
                        2: options.get("team2") or "Team 2",
                    },
                    half=int(options.get("half") or 1),
                    render_video=bool(options.get("render_video", True)),
                    max_seconds=options.get("max_seconds"),
                )
                if options.get("sample_fps"):
                    config.sample_fps = float(options["sample_fps"])

                result = analyse(
                    video_path,
                    config=config,
                    on_progress=on_progress,
                    output_dir=video_path.parent,
                )

                payload = {
                    "events": result.tagger_rows(),
                    "stats": result.stats,
                    "diagnostics": result.diagnostics,
                    "duration_s": result.duration_s,
                }
                self._update(
                    job_id,
                    status="done",
                    stage="done",
                    progress=1.0,
                    message=f"{len(result.events)} events found",
                    finished=datetime.utcnow(),
                    result=json.dumps(payload),
                    output_video=str(result.video_path) if result.video_path else None,
                )

            except KeyboardInterrupt:
                self._update(
                    job_id, status="cancelled", stage="cancelled",
                    message="Analysis cancelled", finished=datetime.utcnow(),
                )
            except Exception as exc:
                self._update(
                    job_id,
                    status="failed",
                    stage="failed",
                    message=str(exc)[:400],
                    error=traceback.format_exc(),
                    finished=datetime.utcnow(),
                )
            finally:
                with self._lock:
                    self._current = None
                self._cancel.clear()


def new_job_id() -> str:
    return str(uuid.uuid4())


def sweep_uploads(upload_dir: Path, keep_days: int = 7) -> int:
    """Delete upload folders older than `keep_days`. Returns how many went.

    A match video is hundreds of megabytes and one is kept per analysis, so
    without this the disk fills quietly. Seven days is long enough to re-import
    a result the analyst forgot to save, short enough to stay bounded.
    """
    import shutil

    if not upload_dir.exists():
        return 0
    cutoff = time.time() - keep_days * 86400
    removed = 0
    for child in upload_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            if child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed
