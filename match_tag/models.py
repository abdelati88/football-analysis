"""Database schema.

`matches` and `events` predate the automatic analysis and still hold every
match tagged by hand, so their existing columns are left exactly as they were.
The new columns are all nullable additions, and `ensure_schema` adds them to an
existing database in place — an installation that has been collecting data for
months must not have to start over to get the new features.
"""
from __future__ import annotations

from datetime import datetime

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text

db = SQLAlchemy()


class Match(db.Model):
    __tablename__ = "matches"

    match_id = db.Column(db.Integer, primary_key=True)
    match_name = db.Column(db.String(200))
    team1 = db.Column(db.String(200))
    team2 = db.Column(db.String(200))
    date_created = db.Column(db.DateTime, server_default=db.func.now())

    # Added with the automatic analysis.
    source = db.Column(db.String(20), default="manual")   # manual | ai | mixed
    video_name = db.Column(db.String(300))
    notes = db.Column(db.Text)

    events = db.relationship(
        "Event", backref="match", cascade="all, delete-orphan", lazy="dynamic"
    )

    def as_dict(self, events_count: int | None = None) -> dict:
        return {
            "match_id": self.match_id,
            "match_name": self.match_name or "Untitled match",
            "team1": self.team1 or "Team 1",
            "team2": self.team2 or "Team 2",
            "date_created": (self.date_created or datetime.utcnow()).isoformat(),
            "source": self.source or "manual",
            "video_name": self.video_name,
            "notes": self.notes,
            "events_count": events_count if events_count is not None else self.events.count(),
        }


class Event(db.Model):
    __tablename__ = "events"

    event_id = db.Column(db.Integer, primary_key=True)
    match_id = db.Column(
        db.Integer, db.ForeignKey("matches.match_id", ondelete="CASCADE"), nullable=False, index=True
    )
    team = db.Column(db.String(100))
    player = db.Column(db.String(200))
    event = db.Column(db.String(100))
    outcome = db.Column(db.String(100))
    mins = db.Column(db.Integer)
    secs = db.Column(db.Integer)
    x = db.Column(db.Float)
    y = db.Column(db.Float)
    x2 = db.Column(db.Float, nullable=True)
    y2 = db.Column(db.Float, nullable=True)

    # Added with the automatic analysis. `half` was previously only in the
    # browser's memory and was lost on save, which made second-half coordinates
    # impossible to reinterpret after loading.
    half = db.Column(db.Integer, default=1)
    source = db.Column(db.String(20), default="manual")   # manual | ai
    confidence = db.Column(db.Float)
    track_id = db.Column(db.Integer)
    frame = db.Column(db.Integer)

    def as_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "Team": self.team,
            "Player": self.player,
            "Event": self.event,
            "Outcome": self.outcome or "",
            "Mins": self.mins or 0,
            "Secs": self.secs or 0,
            "X": self.x,
            "Y": self.y,
            "X2": "" if self.x2 is None else self.x2,
            "Y2": "" if self.y2 is None else self.y2,
            "half": self.half or 1,
            "source": self.source or "manual",
            "confidence": self.confidence,
            "track_id": self.track_id,
            "frame": self.frame,
        }


class AnalysisJob(db.Model):
    """One run of the vision pipeline over one uploaded video."""

    __tablename__ = "analysis_jobs"

    job_id = db.Column(db.String(36), primary_key=True)
    created = db.Column(db.DateTime, server_default=db.func.now())
    finished = db.Column(db.DateTime)
    status = db.Column(db.String(20), default="queued")   # queued|running|done|failed|cancelled
    stage = db.Column(db.String(30), default="queued")
    progress = db.Column(db.Float, default=0.0)
    message = db.Column(db.String(400), default="")
    error = db.Column(db.Text)

    video_name = db.Column(db.String(300))
    video_path = db.Column(db.String(500))
    output_video = db.Column(db.String(500))
    team1 = db.Column(db.String(200))
    team2 = db.Column(db.String(200))
    half = db.Column(db.Integer, default=1)

    # The full result, as JSON text: events, statistics and diagnostics.
    result = db.Column(db.Text)
    match_id = db.Column(db.Integer, db.ForeignKey("matches.match_id"))

    def as_dict(self, include_result: bool = False) -> dict:
        import json

        payload = {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress or 0.0, 3),
            "message": self.message or "",
            "error": self.error,
            "created": (self.created or datetime.utcnow()).isoformat(),
            "finished": self.finished.isoformat() if self.finished else None,
            "video_name": self.video_name,
            "team1": self.team1,
            "team2": self.team2,
            "half": self.half or 1,
            "match_id": self.match_id,
            "has_video": bool(self.output_video),
        }
        if include_result and self.result:
            try:
                payload["result"] = json.loads(self.result)
            except ValueError:
                payload["result"] = None
        return payload


# Columns added after the first release, with the type each needs on SQLite and
# Postgres alike. Kept here so `ensure_schema` and the models cannot drift.
_ADDED_COLUMNS = {
    "matches": {
        "source": "VARCHAR(20)",
        "video_name": "VARCHAR(300)",
        "notes": "TEXT",
    },
    "events": {
        "half": "INTEGER",
        "source": "VARCHAR(20)",
        "confidence": "FLOAT",
        "track_id": "INTEGER",
        "frame": "INTEGER",
    },
}


def ensure_schema() -> list[str]:
    """Create missing tables, then add missing columns to existing ones.

    `db.create_all()` creates tables but never alters one that already exists,
    so a database written by an earlier version would silently lack the new
    columns and every insert naming them would fail.
    """
    db.create_all()

    inspector = inspect(db.engine)
    applied: list[str] = []
    for table, columns in _ADDED_COLUMNS.items():
        if table not in inspector.get_table_names():
            continue
        existing = {c["name"] for c in inspector.get_columns(table)}
        for name, sql_type in columns.items():
            if name in existing:
                continue
            db.session.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}"))
            applied.append(f"{table}.{name}")
    if applied:
        db.session.commit()
    return applied
