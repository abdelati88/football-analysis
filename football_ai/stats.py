"""Match statistics, computed from the tracks and the detected events.

Two families of number live here. Event statistics — passes, shots, tackles —
count what the event builder found. Physical statistics — distance covered, top
speed, average position, heatmaps — come from the tracks directly and do not
depend on any event being recognised at all, which makes them the more robust
half of the output.

The physical side needs care. A tracker that swaps two identities produces an
instantaneous jump of twenty metres, and a naive sum over frame-to-frame
displacement turns that into twenty metres of "running". Every velocity here is
therefore gated against what a footballer can actually do.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from .detect import GOALKEEPER, PLAYER, REFEREE
from .events import (
    CLEARANCE, CROSS, DRIBBLE, GOAL, INTERCEPTION, PASS, SHOT, SHOT_ASSIST,
    SUCCESSFUL, TACKLE, Event,
)
from .pitch import PITCH, PitchConfig

# The fastest a human has run is about 12.4 m/s. Anything faster in the data is
# the tracker changing its mind about who is who.
MAX_PLAYER_SPEED_MS = 12.0
# Sprints are measured over a window rather than between adjacent frames, where
# detection jitter dominates the real movement.
SPEED_WINDOW_S = 0.4


@dataclass
class PlayerStats:
    track_id: int
    team: int
    name: str
    role: str = "player"
    minutes: float = 0.0
    touches: int = 0
    passes: int = 0
    passes_completed: int = 0
    crosses: int = 0
    shots: int = 0
    goals: int = 0
    shot_assists: int = 0
    dribbles: int = 0
    tackles: int = 0
    interceptions: int = 0
    clearances: int = 0
    distance_m: float = 0.0
    top_speed_kmh: float = 0.0
    avg_position: tuple[float, float] | None = None

    @property
    def pass_accuracy(self) -> float:
        return round(100.0 * self.passes_completed / self.passes, 1) if self.passes else 0.0

    def as_dict(self) -> dict:
        d = {
            "track_id": self.track_id,
            "team": self.team,
            "name": self.name,
            "role": self.role,
            "minutes": round(self.minutes, 1),
            "touches": self.touches,
            "passes": self.passes,
            "passes_completed": self.passes_completed,
            "pass_accuracy": self.pass_accuracy,
            "crosses": self.crosses,
            "shots": self.shots,
            "goals": self.goals,
            "shot_assists": self.shot_assists,
            "dribbles": self.dribbles,
            "tackles": self.tackles,
            "interceptions": self.interceptions,
            "clearances": self.clearances,
            "distance_m": round(self.distance_m, 1),
            "distance_km": round(self.distance_m / 1000.0, 2),
            "top_speed_kmh": round(self.top_speed_kmh, 1),
        }
        if self.avg_position is not None:
            d["avg_x"] = round(self.avg_position[0], 1)
            d["avg_y"] = round(self.avg_position[1], 1)
        return d


# ---- physical -----------------------------------------------------------------

def physical_stats(people: pd.DataFrame, fps: float) -> Dict[int, dict]:
    """Distance covered, top speed and average position, per track."""
    if people.empty:
        return {}

    out: Dict[int, dict] = {}
    step = max(int(round(SPEED_WINDOW_S * fps)), 1)

    for track_id, group in people[people["pitch_x"].notna()].groupby("track_id"):
        g = group.sort_values("frame")
        xy = g[["pitch_x", "pitch_y"]].to_numpy()
        times = g["time_s"].to_numpy()
        if len(xy) < 2:
            continue

        # Distance: frame-to-frame, but discard any step that implies an
        # impossible speed rather than letting an identity swap inflate the sum.
        deltas = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        gaps = np.diff(times)
        with np.errstate(divide="ignore", invalid="ignore"):
            speeds = np.where(gaps > 0, deltas / gaps, 0.0)
        credible = np.isfinite(speeds) & (speeds <= MAX_PLAYER_SPEED_MS)
        distance = float(deltas[credible].sum())

        # Top speed over a window, so a single jittery frame cannot set it.
        top = 0.0
        if len(xy) > step:
            window_dist = np.linalg.norm(xy[step:] - xy[:-step], axis=1)
            window_time = times[step:] - times[:-step]
            with np.errstate(divide="ignore", invalid="ignore"):
                window_speed = np.where(window_time > 0, window_dist / window_time, 0.0)
            usable = window_speed[np.isfinite(window_speed) & (window_speed <= MAX_PLAYER_SPEED_MS)]
            if usable.size:
                top = float(usable.max())

        out[int(track_id)] = {
            "distance_m": distance,
            "top_speed_kmh": top * 3.6,
            "minutes": float(times[-1] - times[0]) / 60.0,
            "avg_position": (float(xy[:, 0].mean()), float(xy[:, 1].mean())),
            "frames": int(len(g)),
        }
    return out


def heatmap(
    positions: np.ndarray,
    bins: tuple[int, int] = (12, 8),
    pitch: PitchConfig = PITCH,
) -> List[List[float]]:
    """Normalised occupancy grid over the pitch, as nested lists for JSON."""
    if positions is None or len(positions) == 0:
        return [[0.0] * bins[1] for _ in range(bins[0])]
    hist, _, _ = np.histogram2d(
        positions[:, 0], positions[:, 1],
        bins=bins,
        range=[[0, pitch.length_m], [0, pitch.width_m]],
    )
    peak = hist.max()
    if peak > 0:
        hist = hist / peak
    return [[round(float(v), 3) for v in row] for row in hist]


# ---- events -------------------------------------------------------------------

def _blank(track_id: int, team: int, name: str, role: str) -> PlayerStats:
    return PlayerStats(track_id=track_id, team=team, name=name, role=role)


def player_table(
    events: Sequence[Event],
    people: pd.DataFrame,
    teams: Dict[int, int],
    classes: Dict[int, int],
    fps: float,
    player_names: Dict[int, str] | None = None,
    min_frames: int = 25,
) -> List[PlayerStats]:
    """One row per tracked player, combining event counts and physical output."""
    names = player_names or {}
    physical = physical_stats(people, fps)
    role_of = {PLAYER: "player", GOALKEEPER: "goalkeeper", REFEREE: "referee"}

    # Built from the tracks, not from the physical statistics: distance and
    # speed need pitch coordinates, and a video the calibration model could not
    # read has none. Everything countable — touches, passes, shots — still
    # applies, so the table must survive without geometry.
    seen = people.groupby("track_id").size() if not people.empty else {}

    rows: Dict[int, PlayerStats] = {}
    for track_id, frames in dict(seen).items():
        track_id = int(track_id)
        if frames < min_frames:
            continue
        cls = classes.get(track_id, PLAYER)
        if cls == REFEREE:
            continue
        rows[track_id] = _blank(
            track_id,
            int(teams.get(track_id, 0)),
            names.get(track_id, f"#{track_id}"),
            role_of.get(cls, "player"),
        )
        phys = physical.get(track_id)
        if phys:
            rows[track_id].distance_m = phys["distance_m"]
            rows[track_id].top_speed_kmh = phys["top_speed_kmh"]
            rows[track_id].minutes = phys["minutes"]
            rows[track_id].avg_position = phys["avg_position"]
        else:
            rows[track_id].minutes = float(frames) / fps / 60.0

    for ev in events:
        row = rows.get(int(ev.track_id))
        if row is None:
            continue
        row.touches += 1
        if ev.event == PASS:
            row.passes += 1
            if ev.outcome == SUCCESSFUL:
                row.passes_completed += 1
        elif ev.event == CROSS:
            row.crosses += 1
            row.passes += 1
            if ev.outcome == SUCCESSFUL:
                row.passes_completed += 1
        elif ev.event == SHOT:
            row.shots += 1
            if ev.outcome == GOAL:
                row.goals += 1
        elif ev.event == SHOT_ASSIST:
            row.shot_assists += 1
            row.passes += 1
            row.passes_completed += 1
        elif ev.event == DRIBBLE:
            row.dribbles += 1
        elif ev.event == TACKLE:
            row.tackles += 1
        elif ev.event == INTERCEPTION:
            row.interceptions += 1
        elif ev.event == CLEARANCE:
            row.clearances += 1

    return sorted(rows.values(), key=lambda r: (r.team, -r.touches, r.track_id))


def team_table(
    events: Sequence[Event],
    players: Sequence[PlayerStats],
    possession: Dict[int, float],
    team_names: Dict[int, str],
) -> Dict[int, dict]:
    """Team totals, built by summing the player rows and the event list."""
    teams: Dict[int, dict] = {}
    for team in (1, 2):
        squad = [p for p in players if p.team == team]
        team_events = [e for e in events if e.team == team]

        passes = sum(1 for e in team_events if e.event in (PASS, CROSS, SHOT_ASSIST))
        completed = sum(
            1 for e in team_events
            if (e.event in (PASS, CROSS) and e.outcome == SUCCESSFUL) or e.event == SHOT_ASSIST
        )
        shots = [e for e in team_events if e.event == SHOT]

        teams[team] = {
            "team": team,
            "name": team_names.get(team, f"Team {team}"),
            "possession": possession.get(team, 0.0),
            "passes": passes,
            "passes_completed": completed,
            "pass_accuracy": round(100.0 * completed / passes, 1) if passes else 0.0,
            "shots": len(shots),
            "shots_on_target": sum(1 for e in shots if e.outcome in (GOAL, "Saved")),
            "goals": sum(1 for e in shots if e.outcome == GOAL),
            "crosses": sum(1 for e in team_events if e.event == CROSS),
            "dribbles": sum(1 for e in team_events if e.event == DRIBBLE),
            "tackles": sum(1 for e in team_events if e.event == TACKLE),
            "interceptions": sum(1 for e in team_events if e.event == INTERCEPTION),
            "clearances": sum(1 for e in team_events if e.event == CLEARANCE),
            "distance_km": round(sum(p.distance_m for p in squad) / 1000.0, 2),
            "players": len(squad),
        }
    return teams


def pass_network(events: Sequence[Event], players: Sequence[PlayerStats]) -> Dict[int, dict]:
    """Average positions and completed-pass links, per team.

    A pass is credited to a link by finding whose average position the ball
    arrived nearest to — the event list records where a pass ended, not who
    received it, because the receiver is the *next* event's actor.
    """
    by_team: Dict[int, dict] = {}
    for team in (1, 2):
        squad = [p for p in players if p.team == team and p.avg_position is not None]
        if not squad:
            continue
        nodes = [
            {
                "track_id": p.track_id,
                "name": p.name,
                "x": round(p.avg_position[0], 1),
                "y": round(p.avg_position[1], 1),
                "touches": p.touches,
            }
            for p in squad
        ]
        centres = np.array([[p.avg_position[0], p.avg_position[1]] for p in squad])
        ids = [p.track_id for p in squad]

        links: Dict[tuple[int, int], int] = defaultdict(int)
        for ev in events:
            if ev.team != team or ev.end_m is None:
                continue
            if ev.event not in (PASS, CROSS, SHOT_ASSIST) or ev.outcome not in (SUCCESSFUL, ""):
                continue
            if ev.event in (PASS, CROSS) and ev.outcome != SUCCESSFUL:
                continue
            distances = np.linalg.norm(centres - np.asarray(ev.end_m), axis=1)
            receiver = ids[int(np.argmin(distances))]
            if receiver == ev.track_id:
                continue
            links[(int(ev.track_id), receiver)] += 1

        by_team[team] = {
            "nodes": nodes,
            "links": [
                {"from": a, "to": b, "count": n}
                for (a, b), n in sorted(links.items(), key=lambda kv: -kv[1])
                if n >= 2
            ],
        }
    return by_team


def possession_timeline(
    holder_frame: pd.DataFrame,
    teams: Dict[int, int],
    fps: float,
    bucket_seconds: float = 30.0,
) -> List[dict]:
    """Rolling possession share, one point per bucket."""
    if holder_frame.empty:
        return []
    df = holder_frame[holder_frame["holder"] >= 0].copy()
    if df.empty:
        return []
    df["team"] = df["holder"].map(teams).fillna(0).astype(int)
    df["bucket"] = (df["frame"] / fps // bucket_seconds).astype(int)

    out: List[dict] = []
    for bucket, group in df.groupby("bucket"):
        counts = group["team"].value_counts()
        total = int(counts.sum())
        if not total:
            continue
        out.append(
            {
                "minute": round(bucket * bucket_seconds / 60.0, 1),
                "team1": round(100.0 * int(counts.get(1, 0)) / total, 1),
                "team2": round(100.0 * int(counts.get(2, 0)) / total, 1),
            }
        )
    return out


def summarise(
    events: Sequence[Event],
    people: pd.DataFrame,
    holder_frame: pd.DataFrame,
    teams: Dict[int, int],
    classes: Dict[int, int],
    team_names: Dict[int, str],
    possession: Dict[int, float],
    fps: float,
    player_names: Dict[int, str] | None = None,
    pitch: PitchConfig = PITCH,
) -> dict:
    """The complete statistics payload the web app renders."""
    players = player_table(events, people, teams, classes, fps, player_names)

    heatmaps: Dict[str, List[List[float]]] = {}
    placed = people[people["pitch_x"].notna()]
    for team in (1, 2):
        ids = [t for t, tm in teams.items() if tm == team]
        subset = placed[placed["track_id"].isin(ids)]
        heatmaps[f"team{team}"] = heatmap(subset[["pitch_x", "pitch_y"]].to_numpy(), pitch=pitch)

    return {
        "teams": team_table(events, players, possession, team_names),
        "players": [p.as_dict() for p in players],
        "pass_network": pass_network(events, players),
        "heatmaps": heatmaps,
        "possession_timeline": possession_timeline(holder_frame, teams, fps),
        "totals": {
            "events": len(events),
            "tracked_players": len(players),
            "duration_minutes": round(
                float(people["time_s"].max() - people["time_s"].min()) / 60.0, 1
            ) if not people.empty else 0.0,
        },
    }
