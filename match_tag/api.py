"""HTTP API for saved matches and their events."""
from __future__ import annotations

import csv
import io
import json
from typing import Any

from flask import Blueprint, Response, current_app, jsonify, request
from sqlalchemy import func

from models import Event, Match, db

api = Blueprint("api", __name__, url_prefix="/api")

EVENT_FIELDS = ("Team", "Player", "Event", "Outcome", "Mins", "Secs", "X", "Y", "X2", "Y2")
MAX_EVENTS_PER_MATCH = 20000


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # reject NaN


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def event_from_payload(match_id: int, payload: dict) -> Event:
    return Event(
        match_id=match_id,
        team=(payload.get("Team") or "")[:100],
        player=(payload.get("Player") or "")[:200],
        event=(payload.get("Event") or "")[:100],
        outcome=(payload.get("Outcome") or "")[:100],
        mins=_to_int(payload.get("Mins")),
        secs=_to_int(payload.get("Secs")),
        x=_to_float(payload.get("X")),
        y=_to_float(payload.get("Y")),
        x2=_to_float(payload.get("X2")),
        y2=_to_float(payload.get("Y2")),
        half=_to_int(payload.get("half"), 1) or 1,
        source=(payload.get("source") or "manual")[:20],
        confidence=_to_float(payload.get("confidence")),
        track_id=_to_int(payload.get("track_id"), 0) or None,
        frame=_to_int(payload.get("frame"), 0) or None,
    )


@api.post("/matches")
def create_match():
    """Save a match and all of its events."""
    data = request.get_json(silent=True) or {}
    events = data.get("events") or []
    if not isinstance(events, list):
        return jsonify({"error": "events must be a list"}), 400
    if len(events) > MAX_EVENTS_PER_MATCH:
        return jsonify({"error": f"a match may hold at most {MAX_EVENTS_PER_MATCH} events"}), 400

    sources = {(e.get("source") or "manual") for e in events}
    match = Match(
        match_name=(data.get("matchName") or "Untitled match")[:200],
        team1=(data.get("team1") or "Team 1")[:200],
        team2=(data.get("team2") or "Team 2")[:200],
        video_name=(data.get("videoName") or None),
        notes=data.get("notes"),
        source="mixed" if len(sources) > 1 else (sources.pop() if sources else "manual"),
    )
    db.session.add(match)
    db.session.flush()

    db.session.bulk_save_objects(
        [event_from_payload(match.match_id, e) for e in events if isinstance(e, dict)]
    )
    db.session.commit()
    return jsonify({"success": True, "match_id": match.match_id}), 201


@api.put("/matches/<int:match_id>")
def replace_match(match_id: int):
    """Overwrite a saved match with the current state of the tagger."""
    match = db.session.get(Match, match_id)
    if match is None:
        return jsonify({"error": "no such match"}), 404

    data = request.get_json(silent=True) or {}
    events = data.get("events") or []
    if len(events) > MAX_EVENTS_PER_MATCH:
        return jsonify({"error": f"a match may hold at most {MAX_EVENTS_PER_MATCH} events"}), 400

    if data.get("matchName"):
        match.match_name = data["matchName"][:200]
    if data.get("team1"):
        match.team1 = data["team1"][:200]
    if data.get("team2"):
        match.team2 = data["team2"][:200]
    match.notes = data.get("notes", match.notes)

    Event.query.filter_by(match_id=match_id).delete()
    db.session.bulk_save_objects(
        [event_from_payload(match_id, e) for e in events if isinstance(e, dict)]
    )
    db.session.commit()
    return jsonify({"success": True, "match_id": match_id, "events": len(events)})


@api.post("/client-error")
def client_error():
    """Record a JavaScript error that happened in somebody's browser.

    The interface is the half of this program that cannot be debugged from
    here. A script that throws during start-up leaves a page that looks
    comprehensively broken while saying nothing to anyone who is not standing
    in front of it with the console open — which is never the person who hits
    it. This gives the page a way to say what went wrong.
    """
    payload = request.get_json(silent=True) or {}
    current_app.logger.error(
        "client error: %s\n  at %s:%s:%s\n%s",
        str(payload.get("message"))[:500],
        str(payload.get("source"))[:200],
        payload.get("line"), payload.get("column"),
        str(payload.get("stack"))[:2000],
    )
    return jsonify({"logged": True})


@api.get("/matches")
def list_matches():
    limit = min(_to_int(request.args.get("limit"), 100), 500)
    counts = dict(
        db.session.query(Event.match_id, func.count(Event.event_id))
        .group_by(Event.match_id)
        .all()
    )
    rows = Match.query.order_by(Match.date_created.desc()).limit(limit).all()
    return jsonify([m.as_dict(events_count=counts.get(m.match_id, 0)) for m in rows])


@api.get("/matches/<int:match_id>")
def get_match(match_id: int):
    match = db.session.get(Match, match_id)
    if match is None:
        return jsonify({"error": "no such match"}), 404
    events = Event.query.filter_by(match_id=match_id).order_by(
        Event.mins, Event.secs, Event.event_id
    ).all()
    payload = match.as_dict(events_count=len(events))
    payload["events"] = [e.as_dict() for e in events]
    return jsonify(payload)


@api.delete("/matches/<int:match_id>")
def delete_match(match_id: int):
    match = db.session.get(Match, match_id)
    if match is None:
        return jsonify({"error": "no such match"}), 404
    Event.query.filter_by(match_id=match_id).delete()
    db.session.delete(match)
    db.session.commit()
    return jsonify({"success": True})


@api.get("/matches/<int:match_id>/events")
def get_events(match_id: int):
    if db.session.get(Match, match_id) is None:
        return jsonify({"error": "no such match"}), 404
    rows = Event.query.filter_by(match_id=match_id).order_by(
        Event.mins, Event.secs, Event.event_id
    ).all()
    return jsonify([e.as_dict() for e in rows])


@api.get("/matches/<int:match_id>/export.csv")
def export_csv(match_id: int):
    match = db.session.get(Match, match_id)
    if match is None:
        return jsonify({"error": "no such match"}), 404
    rows = Event.query.filter_by(match_id=match_id).order_by(
        Event.mins, Event.secs, Event.event_id
    ).all()

    # Written with the csv module rather than string concatenation: a team or
    # player name containing a comma or a quote would otherwise silently shift
    # every following column.
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([*EVENT_FIELDS, "Half", "Source", "Confidence"])
    for r in rows:
        d = r.as_dict()
        writer.writerow(
            [d[f] for f in EVENT_FIELDS]
            + [d["half"], d["source"], "" if d["confidence"] is None else round(d["confidence"], 2)]
        )

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in (match.match_name or "match"))
    return Response(
        buffer.getvalue(),
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}_{match_id}.csv"'},
    )


@api.get("/matches/<int:match_id>/export.json")
def export_json(match_id: int):
    match = db.session.get(Match, match_id)
    if match is None:
        return jsonify({"error": "no such match"}), 404
    rows = Event.query.filter_by(match_id=match_id).order_by(
        Event.mins, Event.secs, Event.event_id
    ).all()
    payload = match.as_dict(events_count=len(rows))
    payload["events"] = [e.as_dict() for e in rows]
    return Response(
        json.dumps(payload, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": f'attachment; filename="match_{match_id}.json"'},
    )
