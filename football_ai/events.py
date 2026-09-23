"""Turning possession into football.

A list of "player 7 held the ball from frame 400 to 412" is not analysis. This
module reads the transitions between those spells and names them: a change of
possession inside a team is a pass, a change between teams is a tackle or an
interception, possession ending with the ball travelling at a goal is a shot.

The vocabulary is deliberately the one the manual tagger already uses — Pass,
Cross, Shot, Dribble, Tackle, Interception, Clearance, Shot Assist — with the
same outcomes and the same 120 x 80 coordinates. An automatically detected
event and a hand-tagged one are the same kind of object, so the analyst can
correct the machine's work in place instead of choosing between the two.

Every event carries a `confidence`. Nothing here is certain: it is geometry and
timing applied to imperfect detections, and the interface should be able to
show the analyst which calls to check first.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from .pitch import (
    PITCH, TAGGER_LENGTH, TAGGER_WIDTH, PitchConfig, metres_to_tagger, mirror_metres,
)
from .possession import Spell

# The event vocabulary, matching the tagger's own buttons.
PASS = "Pass"
CROSS = "Cross"
SHOT = "Shot"
DRIBBLE = "Dribble"
TACKLE = "Tackle"
INTERCEPTION = "Interception"
CLEARANCE = "Clearance"
SHOT_ASSIST = "Shot Assist"

SUCCESSFUL, UNSUCCESSFUL, FAILED = "Successful", "Unsuccessful", "Failed"
GOAL, SAVED, OFF_TARGET, BLOCKED = "Goal", "Saved", "Off Target", "Blocked"


@dataclass
class EventConfig:
    # A "pass" that travels less than this is a touch or a tracking wobble.
    min_pass_distance_m: float = 3.0
    # Beyond this the two spells are unrelated; the ball went out of play or was
    # lost by the tracker, and inventing a pass between them would be fiction.
    max_flight_seconds: float = 8.0
    long_ball_m: float = 32.0
    # Above this, the two possessions are not connected by a kick. The pitch is
    # 105 m long, and the longest ball anyone actually strikes is a goalkeeper
    # clearing his line; an outfield player reaching sixty metres has hit the
    # ball about as far as the game allows. Past that the far likelier story is
    # that the tracker lost the ball and re-found it somewhere else entirely,
    # and drawing a pass between the two is the same fiction that
    # `max_flight_seconds` already refuses on the time axis.
    #
    # Found by a 74.7 m "pass" in the last second of a clip — which the
    # evidence score had already marked the weakest event of the run, but which
    # was recorded anyway because nothing was checking the distance.
    max_pass_distance_m: float = 60.0
    max_keeper_distance_m: float = 85.0
    # A turnover with the two players this close, resolved this fast, is a duel
    # won on the ground rather than a pass that was read and intercepted.
    tackle_distance_m: float = 3.5
    tackle_seconds: float = 1.2
    carry_distance_m: float = 4.0
    # Shots: how far out one may be taken from, and how much wider than the
    # posts a goal-bound line may point and still count as an attempt.
    max_shot_distance_m: float = 35.0
    shot_target_margin_m: float = 3.5
    min_shot_speed_ms: float = 7.0
    # A ball merely *pointed* at goal is usually a pass that happens to be
    # aimed forward. An attempt has to arrive: the tracked ball must get this
    # close to the goal line, or be gathered by the opposing keeper.
    shot_reach_m: float = 14.0
    # Clearances come from deep and go long.
    clearance_zone_frac: float = 0.34
    clearance_distance_m: float = 22.0
    # How much of the confidence is the classification rule's own reliability
    # and how much is the evidence for this particular event.
    #
    # The rule prior alone was the whole number once, and it made the column
    # dishonest: a pass whose ball was tracked cleanly from foot to foot and a
    # pass whose ball vanished and was interpolated across the flight scored
    # identically, though one is an observation and the other a reconstruction.
    # The floor is what remains when the evidence is at its worst, so a rule
    # that is usually right is never reported as a coin flip merely because one
    # instance was poorly seen.
    evidence_floor: float = 0.55
    # Detection confidence at or above this counts as a clean sighting; the
    # ball detector's own scores sit around 0.7 on a good frame.
    ball_conf_reference: float = 0.7

    # A shot assist is the pass that set up a shot taken within this window.
    shot_assist_seconds: float = 5.0


@dataclass
class Event:
    """One detected event, in both metric and tagger coordinates."""

    team: int
    team_name: str
    player: str
    track_id: int
    event: str
    outcome: str
    frame: int
    time_s: float
    half: int
    # Pitch metres — what the statistics are computed from.
    start_m: tuple[float, float] = (0.0, 0.0)
    end_m: tuple[float, float] | None = None
    confidence: float = 0.5
    note: str = ""

    # ---- the tagger's coordinate system ------------------------------------

    @property
    def mins(self) -> int:
        return int(self.time_s // 60)

    @property
    def secs(self) -> int:
        return int(self.time_s % 60)

    def _to_tagger(self, point: tuple[float, float]) -> tuple[float, float]:
        """Pitch metres -> the tagger's grid, normalised for the half.

        The manual tagger has always recorded second-half events rotated 180
        degrees about the centre spot, so that a team's attacks point the same
        way in the data whichever end they were kicking towards. Detected
        events have to follow the same convention or the two would be drawn in
        different halves of the same pitch.

        Coordinates are clamped to the grid. A ball genuinely does go over the
        line, and calibration error can push a position further out still, but
        the tagger's canvas *is* the pitch — anything outside it simply cannot
        be drawn, and a hand-tagged click could never land there either.
        """
        metres = mirror_metres([point])[0] if self.half == 2 else point
        x, y = metres_to_tagger([metres])[0]
        return (
            float(min(max(x, 0.0), TAGGER_LENGTH)),
            float(min(max(y, 0.0), TAGGER_WIDTH)),
        )

    def as_tagger_row(self) -> dict:
        """The exact shape the tagger stores and draws."""
        sx, sy = self._to_tagger(self.start_m)
        row = {
            "Team": self.team_name,
            "Player": self.player,
            "Event": self.event,
            "Outcome": self.outcome,
            "Mins": self.mins,
            "Secs": self.secs,
            "X": round(float(sx), 1),
            "Y": round(float(sy), 1),
            "X2": "",
            "Y2": "",
            "half": self.half,
            "source": "ai",
            "confidence": round(float(self.confidence), 2),
            "track_id": self.track_id,
            "frame": self.frame,
        }
        if self.end_m is not None:
            ex, ey = self._to_tagger(self.end_m)
            row["X2"] = round(float(ex), 1)
            row["Y2"] = round(float(ey), 1)
        return row


# ---- attacking direction ----------------------------------------------------

def infer_attacking_directions(
    people: pd.DataFrame,
    teams: Dict[int, int],
    classes: Dict[int, int],
    pitch: PitchConfig = PITCH,
) -> Dict[int, int]:
    """Which goal each team attacks: +1 towards x = length, -1 towards x = 0.

    A goalkeeper is the clearest signal on the pitch — he stands in front of the
    goal his team defends, and a team attacks the other one. Where no keeper was
    detected, the fallback is the team's two deepest players, who occupy the
    same end for the same reason.
    """
    if people.empty:
        return {}

    placed = people[people["pitch_x"].notna()].copy()
    if placed.empty:
        return {}
    placed["team"] = placed["track_id"].map(teams).fillna(0).astype(int)
    placed["cls"] = placed["track_id"].map(classes).fillna(placed["cls"]).astype(int)

    from .detect import GOALKEEPER

    half_length = pitch.length_m / 2.0
    directions: Dict[int, int] = {}

    for team in (1, 2):
        squad = placed[placed["team"] == team]
        if squad.empty:
            continue
        keepers = squad[squad["cls"] == GOALKEEPER]
        if len(keepers) >= 10:
            own_goal_x = float(keepers["pitch_x"].median())
        else:
            # Deepest players by track: take each track's median x, then the two
            # extreme ones, and see which end they cluster at.
            per_track = squad.groupby("track_id")["pitch_x"].median()
            if per_track.empty:
                continue
            low = per_track.nsmallest(2).mean()
            high = per_track.nlargest(2).mean()
            own_goal_x = low if abs(low - 0.0) < abs(high - pitch.length_m) else high
        directions[team] = 1 if own_goal_x < half_length else -1

    # Two teams cannot attack the same goal; if the signals disagreed, trust the
    # one whose own-goal estimate was more extreme by flipping the other.
    if len(directions) == 2 and directions[1] == directions[2]:
        directions[2] = -directions[1]
    return directions


# ---- shot geometry ----------------------------------------------------------

def _goal_line_x(direction: int, pitch: PitchConfig) -> float:
    return pitch.length_m if direction > 0 else 0.0


def _crosses_goal_mouth(
    origin: np.ndarray,
    velocity: np.ndarray,
    direction: int,
    pitch: PitchConfig,
    margin: float,
) -> tuple[bool, float] | tuple[bool, None]:
    """Does a ball leaving `origin` along `velocity` reach the goal mouth?

    Returns `(on_target_line, y_at_goal_line)`. `margin` widens the mouth to
    absorb tracking error and to include attempts that miss narrowly — those
    are still shots.
    """
    goal_x = _goal_line_x(direction, pitch)
    vx = float(velocity[0])
    if abs(vx) < 1e-6 or np.sign(vx) != np.sign(direction):
        return False, None
    t = (goal_x - float(origin[0])) / vx
    if t <= 0:
        return False, None
    y_at_goal = float(origin[1] + velocity[1] * t)
    half_mouth = pitch.goal_width / 200.0 + margin
    centre_y = pitch.width_m / 2.0
    return abs(y_at_goal - centre_y) <= half_mouth, y_at_goal


def _classify_shot_outcome(
    ball_path: np.ndarray,
    next_spell: Spell | None,
    direction: int,
    y_at_goal: float | None,
    pitch: PitchConfig,
    config: EventConfig,
) -> tuple[str, float]:
    """Name the outcome of an attempt, with a confidence in that name."""
    goal_x = _goal_line_x(direction, pitch)
    half_mouth = pitch.goal_width / 200.0
    centre_y = pitch.width_m / 2.0

    # Did the tracked ball actually end up past the line, between the posts?
    if len(ball_path):
        beyond = (
            ball_path[:, 0] >= goal_x - 0.5 if direction > 0 else ball_path[:, 0] <= goal_x + 0.5
        )
        if beyond.any():
            crossing_y = float(ball_path[beyond][0, 1])
            if abs(crossing_y - centre_y) <= half_mouth:
                return GOAL, 0.75
            return OFF_TARGET, 0.7

    if next_spell is not None:
        if next_spell.is_goalkeeper:
            return SAVED, 0.65
        # An opponent taking the ball right in front of the shooter, close to
        # goal, is a block rather than a clean interception.
        distance_to_goal = abs(float(next_spell.start_xy[0]) - goal_x)
        if distance_to_goal < 18.0:
            return BLOCKED, 0.5

    if y_at_goal is not None and abs(y_at_goal - centre_y) <= half_mouth:
        # Pointed in, but we never saw it arrive — most often a save we lost
        # sight of behind the keeper.
        return SAVED, 0.4
    return OFF_TARGET, 0.45


def _trim_flight(
    path: np.ndarray, frames: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cut a ball path at the first point where it doubles back.

    Everything after a reversal belongs to whatever happened next — a save
    parried clear, a rebound, a keeper's throw — not to the flight that started
    it. Judging a shot on a path that includes the clearance would have it
    travelling away from the goal it was struck at.
    """
    if len(path) < 3:
        return path, frames
    heading = path[1] - path[0]
    if np.linalg.norm(heading) < 1e-6:
        return path, frames
    heading = heading / np.linalg.norm(heading)
    for i in range(2, len(path)):
        step = path[i] - path[i - 1]
        if np.linalg.norm(step) < 1e-6:
            continue
        if float(np.dot(step / np.linalg.norm(step), heading)) < -0.3:
            return path[:i], frames[:i]
    return path, frames


# ---- the main derivation ----------------------------------------------------

class EventBuilder:
    def __init__(
        self,
        team_names: Dict[int, str],
        directions: Dict[int, int],
        player_names: Dict[int, str] | None = None,
        pitch: PitchConfig = PITCH,
        config: EventConfig | None = None,
        half: int = 1,
        fps: float = 25.0,
    ):
        self.team_names = team_names
        self.directions = directions
        self.player_names = player_names or {}
        self.pitch = pitch
        self.config = config or EventConfig()
        self.half = half
        self.fps = fps or 25.0

    # -- helpers --------------------------------------------------------------

    def _player(self, track_id: int) -> str:
        return self.player_names.get(int(track_id), f"#{int(track_id)}")

    def _team_name(self, team: int) -> str:
        return self.team_names.get(team, f"Team {team}" if team else "Unknown")

    def _weigh(self, prior: float, f0: int, f1: int) -> float:
        """Temper a rule's own reliability with the evidence for this instance."""
        floor = self.config.evidence_floor
        return round(prior * (floor + (1.0 - floor) * self._evidence(f0, f1)), 3)

    def _evidence(self, f0: int, f1: int) -> float:
        """How well was the ball actually seen over this stretch, in [0, 1]?

        Every event here is inferred from where the ball was, so the quality of
        that answer is the quality of the event. Two things are asked of the
        stretch: what share of its frames held a real sighting rather than a
        value the interpolation invented, and how confident the detector was in
        the sightings it did make.

        They are multiplied rather than averaged, because both must hold. A
        stretch seen in every frame at low confidence and a stretch seen in a
        third of its frames at high confidence are both weak evidence, and an
        average would quietly rescue each of them with the other's strength.
        """
        ball = getattr(self, "_ball_frame", None)
        if ball is None or ball.empty:
            return 1.0
        window = ball[(ball["frame"] >= f0) & (ball["frame"] <= f1)]
        if window.empty:
            return 0.0

        # Whichever position column this frame carries. A caller may hand over
        # a reduced table — the tests do — and a missing column is a reason to
        # judge on what is present, not to fail.
        position = next(
            (c for c in ("img_x", "pitch_x") if c in window.columns), None
        )
        if position is None:
            return 1.0

        seen = window[position].notna()
        if "interpolated" in window.columns:
            # Interpolated rows carry a position but no observation behind it.
            seen = seen & ~window["interpolated"].fillna(False)
        if not seen.any():
            return 0.0
        share = float(seen.mean())

        if "conf" not in window.columns:
            return max(0.0, min(1.0, share))
        strength = float(
            (window.loc[seen, "conf"] / self.config.ball_conf_reference).clip(0.0, 1.0).mean()
        )
        return max(0.0, min(1.0, share * strength))

    def _make(self, spell: Spell, event: str, outcome: str, start, end, conf: float, note: str = "") -> Event:
        conf = self._weigh(conf, spell.start_frame, spell.end_frame)
        return Event(
            team=spell.team,
            team_name=self._team_name(spell.team),
            player=self._player(spell.track_id),
            track_id=int(spell.track_id),
            event=event,
            outcome=outcome,
            frame=spell.end_frame,
            time_s=spell.end_time,
            half=self.half,
            start_m=(float(start[0]), float(start[1])),
            end_m=None if end is None else (float(end[0]), float(end[1])),
            confidence=conf,
            note=note,
        )

    def _ball_path(
        self, ball: pd.DataFrame, f0: int, f1: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ball positions in metres between two frames, with their frame numbers."""
        window = ball[(ball["frame"] >= f0) & (ball["frame"] <= f1)]
        window = window[window["pitch_x"].notna()]
        if window.empty:
            return np.empty((0, 2)), np.empty(0, dtype=int)
        return (
            window[["pitch_x", "pitch_y"]].to_numpy(),
            window["frame"].to_numpy().astype(int),
        )

    def _is_shot(
        self, spell: Spell, path: np.ndarray, flight: float, nxt: Spell | None
    ) -> tuple[bool, float | None]:
        direction = self.directions.get(spell.team)
        if direction is None or len(path) < 2:
            return False, None

        origin = path[0]
        goal_x = _goal_line_x(direction, self.pitch)
        if abs(float(origin[0]) - goal_x) > self.config.max_shot_distance_m:
            return False, None

        velocity = path[-1] - path[0]
        travelled = float(np.linalg.norm(velocity))
        if travelled < 2.0 or flight <= 0:
            return False, None
        if travelled / flight < self.config.min_shot_speed_ms:
            return False, None

        # Aimed at the goal is not enough — a forward pass in the final third
        # is aimed at the goal too. The ball has to end up there, or in the
        # opposing keeper's hands.
        arrived = abs(float(path[-1][0]) - goal_x) <= self.config.shot_reach_m
        gathered_by_keeper = (
            nxt is not None and nxt.is_goalkeeper and nxt.team != spell.team
        )
        if not (arrived or gathered_by_keeper):
            return False, None

        on_line, y_at_goal = _crosses_goal_mouth(
            origin, velocity, direction, self.pitch, self.config.shot_target_margin_m
        )
        return on_line, y_at_goal

    def _is_cross(self, spell: Spell, start: np.ndarray, end: np.ndarray) -> bool:
        direction = self.directions.get(spell.team)
        if direction is None:
            return False
        L = self.pitch.length_m
        in_final_third = (start[0] > 2 * L / 3) if direction > 0 else (start[0] < L / 3)
        _, y0, _, y1 = self.pitch.penalty_box_m("left")
        from_wide = start[1] < y0 or start[1] > y1
        side = "right" if direction > 0 else "left"
        return bool(in_final_third and from_wide and self.pitch.in_penalty_box(end, side))

    def _is_clearance(self, spell: Spell, start: np.ndarray, distance: float) -> bool:
        direction = self.directions.get(spell.team)
        if direction is None:
            return False
        L = self.pitch.length_m
        own_third = (start[0] < L * self.config.clearance_zone_frac) if direction > 0 \
            else (start[0] > L * (1 - self.config.clearance_zone_frac))
        return bool(own_third and distance >= self.config.clearance_distance_m)

    # -- the pass over the spells ---------------------------------------------

    def build(self, spells: Sequence[Spell], ball: pd.DataFrame) -> List[Event]:
        events: List[Event] = []
        cfg = self.config
        self._ball_frame = ball

        for i, spell in enumerate(spells):
            nxt = spells[i + 1] if i + 1 < len(spells) else None

            # --- carried the ball themselves --------------------------------
            if spell.carry_distance >= cfg.carry_distance_m and spell.duration > 0.4:
                kept = nxt is not None and nxt.team == spell.team
                events.append(
                    self._make(
                        spell, DRIBBLE, SUCCESSFUL if kept else FAILED,
                        spell.start_xy, spell.end_xy,
                        conf=0.55,
                        note=f"carried {spell.carry_distance:.1f} m",
                    )
                )

            start = np.asarray(spell.release_point(), dtype=float)

            # --- the ball's flight after this spell -------------------------
            # Bounded, and evaluated even when there is no next spell, so the
            # last touch of a clip can still be recognised as a shot.
            #
            # The window normally stops where the next player takes over. The
            # one exception is the opposing goalkeeper: a shot that beats him is
            # still crossing the line at the moment he is recorded as nearest to
            # the ball, so cutting there would turn every goal into a save. In
            # that case only, the window runs a little further, and
            # `_trim_flight` ends the path where the ball actually turns round
            # so nothing he then does with it is read as part of the shot.
            max_flight_frames = int(cfg.max_flight_seconds * self.fps)
            window_end = spell.contact_end + max_flight_frames
            if nxt is not None:
                keeper_collects = nxt.is_goalkeeper and nxt.team != spell.team
                tail = int(0.7 * self.fps) if keeper_collects else 0
                window_end = min(nxt.start_frame + tail, window_end)
            path, path_frames = self._ball_path(ball, spell.contact_end, window_end)
            path, path_frames = _trim_flight(path, path_frames)

            # Time the flight by the ball's own first and last sighting. Using
            # the spell boundaries instead would divide a real distance by a
            # window that is mostly the ball sitting still, and every shot would
            # read as too slow to be one.
            if len(path_frames) >= 2:
                shot_flight = max((path_frames[-1] - path_frames[0]) / self.fps, 1e-3)
            else:
                shot_flight = 1e-3

            # --- an attempt on goal -----------------------------------------
            is_shot, y_at_goal = self._is_shot(spell, path, shot_flight, nxt)
            if is_shot:
                direction = self.directions[spell.team]
                outcome, conf = _classify_shot_outcome(
                    path, nxt, direction, y_at_goal, self.pitch, cfg
                )
                goal_x = _goal_line_x(direction, self.pitch)
                events.append(
                    self._make(
                        spell, SHOT, outcome, start,
                        (goal_x, y_at_goal if y_at_goal is not None else self.pitch.width_m / 2),
                        conf=conf,
                    )
                )
                continue

            if nxt is None:
                continue

            flight = max(nxt.start_time - spell.end_time, 1e-3)
            if flight > cfg.max_flight_seconds:
                # The thread of play was lost between these two spells: the ball
                # went out, or the tracker dropped it for too long.
                continue

            end = np.asarray(nxt.receive_point(), dtype=float)
            distance = float(np.linalg.norm(end - start))
            same_team = spell.team and nxt.team and spell.team == nxt.team

            # --- lost in a duel ---------------------------------------------
            # Checked before the distance guard: a tackle is precisely the case
            # where the ball does not travel.
            if not same_team:
                proximity = float(
                    np.linalg.norm(np.asarray(nxt.start_xy) - np.asarray(spell.end_xy))
                )
                if proximity <= cfg.tackle_distance_m and flight <= cfg.tackle_seconds:
                    events.append(
                        Event(
                            team=nxt.team,
                            team_name=self._team_name(nxt.team),
                            player=self._player(nxt.track_id),
                            track_id=int(nxt.track_id),
                            event=TACKLE,
                            outcome=SUCCESSFUL,
                            frame=nxt.start_frame,
                            time_s=nxt.start_time,
                            half=self.half,
                            start_m=(float(nxt.start_xy[0]), float(nxt.start_xy[1])),
                            end_m=None,
                            confidence=self._weigh(0.55, nxt.start_frame, nxt.end_frame),
                            note=f"won the ball {proximity:.1f} m from the carrier",
                        )
                    )
                    continue

            if distance < cfg.min_pass_distance_m:
                # Ball barely moved and nobody won it in a duel: a scuffed
                # touch, or the tracker swapping two players standing together.
                continue

            reach = (
                cfg.max_keeper_distance_m if spell.is_goalkeeper
                else cfg.max_pass_distance_m
            )
            if distance > reach:
                # Further than the ball can have been kicked. Something was
                # lost between these two possessions; say nothing rather than
                # invent the pass that would have to have joined them.
                continue

            # --- kept in the team -------------------------------------------
            if same_team:
                if self._is_cross(spell, start, end):
                    kind, conf = CROSS, 0.6
                elif distance >= cfg.long_ball_m:
                    kind, conf = PASS, 0.6
                else:
                    kind, conf = PASS, 0.75
                events.append(self._make(spell, kind, SUCCESSFUL, start, end, conf))
                continue

            # --- lost to the opposition -------------------------------------
            if self._is_clearance(spell, start, distance):
                events.append(self._make(spell, CLEARANCE, "", start, end, conf=0.5))
                continue

            events.append(
                self._make(spell, CROSS if self._is_cross(spell, start, end) else PASS,
                           UNSUCCESSFUL, start, end, conf=0.65)
            )
            events.append(
                Event(
                    team=nxt.team,
                    team_name=self._team_name(nxt.team),
                    player=self._player(nxt.track_id),
                    track_id=int(nxt.track_id),
                    event=INTERCEPTION,
                    outcome="",
                    frame=nxt.start_frame,
                    time_s=nxt.start_time,
                    half=self.half,
                    start_m=(float(nxt.start_xy[0]), float(nxt.start_xy[1])),
                    end_m=None,
                    confidence=self._weigh(0.55, nxt.start_frame, nxt.end_frame),
                )
            )

        return self._add_shot_assists(events)

    def _add_shot_assists(self, events: List[Event]) -> List[Event]:
        """Relabel the successful pass that immediately set up a shot.

        Only the pass whose receiver is the shooter counts, and only within a
        few seconds — otherwise every pass in a passage of play ending in a
        shot would be credited with creating it.
        """
        window = self.config.shot_assist_seconds
        shots = [e for e in events if e.event == SHOT]
        if not shots:
            return events

        for shot in shots:
            best: Event | None = None
            for candidate in events:
                if candidate.event != PASS or candidate.outcome != SUCCESSFUL:
                    continue
                if candidate.team != shot.team or candidate.end_m is None:
                    continue
                gap = shot.time_s - candidate.time_s
                if not (0 < gap <= window):
                    continue
                # The pass must have arrived where the shot was struck.
                reached = float(
                    np.linalg.norm(np.asarray(candidate.end_m) - np.asarray(shot.start_m))
                )
                if reached > 6.0:
                    continue
                if best is None or candidate.time_s > best.time_s:
                    best = candidate
            if best is not None:
                best.event = SHOT_ASSIST
                best.confidence = min(best.confidence, 0.6)
        return events


def events_to_rows(events: Iterable[Event]) -> List[dict]:
    return [e.as_tagger_row() for e in events]
