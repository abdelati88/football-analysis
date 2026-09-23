"""Who has the ball, and for how long.

Every event the analysis produces is derived from this: a pass is a change of
possession within a team, a tackle is a change between teams, a shot is
possession ending at the goal. Get possession wrong and everything downstream
is wrong, so this module is deliberately conservative — it would rather report
no possession than the wrong player's.

Two ideas do most of the work:

  * Distance is measured on the pitch, in metres, not in pixels. Two players
    twenty pixels apart are standing beside each other at the near touchline and
    fifteen metres apart at the far one; only metres are comparable across a
    frame.
  * A single frame's nearest player is noise. Possession has to be *held* for a
    fraction of a second before it counts, and it survives brief interruptions,
    so a defender's leg passing in front of the ball does not register as a
    change of possession.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .detect import GOALKEEPER, PLAYER, REFEREE


@dataclass
class PossessionConfig:
    # How close a player's feet must be to the ball, in metres, to be a
    # candidate for holding it. Roughly the reach of a leg plus tracking error.
    max_distance_m: float = 2.4
    # Pixel fallback for frames with no homography, as a fraction of frame width.
    max_distance_px_frac: float = 0.045
    # Frames a candidate must lead for before possession switches to them.
    # Four rather than three: at the ~12 fps the pipeline analyses, a ball
    # flying past a defender is within touching distance of them for two or
    # three frames, and would otherwise be recorded as that defender briefly
    # winning the ball.
    confirm_frames: int = 4
    # Frames of "nobody near the ball" tolerated before a spell is closed —
    # this is the ball in flight during a pass, which must not end possession
    # prematurely nor be attributed to whoever it flies past.
    release_frames: int = 4
    # Spells shorter than this are touches, not possession, and are dropped.
    min_spell_frames: int = 2
    # Frames an *opponent* must lead for before the ball is called a turnover.
    # Deliberately longer than `confirm_frames`: a team-mate receiving the ball
    # and an opponent taking it are not equally likely, and should not need
    # equal evidence. Measured against six minutes of hand-tagged play, the
    # shipped code called 121 turnovers where a person saw perhaps twenty, and
    # every spurious one manufactures an interception out of two players
    # standing near the same ball.
    steal_frames: int = 4
    # A player who loses the ball to *nobody* and has it again within this many
    # frames never lost it. Without this the gap becomes two possessions with a
    # fabricated event between them.
    regain_frames: int = 2
    # Distance a player must cover while holding the ball for it to be a carry.
    carry_distance_m: float = 4.0


@dataclass
class Spell:
    """One uninterrupted period of a single player holding the ball."""

    track_id: int
    team: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    start_xy: np.ndarray            # pitch metres, player's feet
    end_xy: np.ndarray
    ball_start_xy: np.ndarray | None = None
    ball_end_xy: np.ndarray | None = None
    is_goalkeeper: bool = False
    # The frames the player was actually seen in during this spell. A spell can
    # run past the last sighting — possession survives a short gap — and the
    # ball is already in flight by then, so these are the frames to read the
    # ball's position at. Default to the spell's own bounds when a caller does
    # not distinguish them.
    contact_start: int | None = None
    contact_end: int | None = None

    def __post_init__(self) -> None:
        if self.contact_start is None:
            self.contact_start = self.start_frame
        if self.contact_end is None:
            self.contact_end = self.end_frame

    @property
    def duration(self) -> float:
        return max(self.end_time - self.start_time, 0.0)

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def carry_distance(self) -> float:
        return float(np.linalg.norm(self.end_xy - self.start_xy))

    def release_point(self) -> np.ndarray:
        """Where the ball left this player — the best origin for a pass."""
        return self.ball_end_xy if self.ball_end_xy is not None else self.end_xy

    def receive_point(self) -> np.ndarray:
        return self.ball_start_xy if self.ball_start_xy is not None else self.start_xy


def assign_ball_holder(
    people: pd.DataFrame,
    ball: pd.DataFrame,
    teams: Dict[int, int],
    classes: Dict[int, int],
    config: PossessionConfig,
    frame_width: int,
) -> pd.DataFrame:
    """Per frame, the nearest eligible player to the ball and how far away.

    Returns one row per ball frame with `holder`, `holder_team` and `distance_m`
    (or `distance_px` when the frame had no homography). `holder` is -1 when
    nobody is close enough.
    """
    if ball.empty or people.empty:
        return pd.DataFrame(columns=["frame", "holder", "holder_team", "distance"])

    # Referees are on the pitch and often near the ball; they never hold it.
    eligible = people[people["cls"] != REFEREE]
    by_frame = {int(f): g for f, g in eligible.groupby("frame")}

    px_limit = frame_width * config.max_distance_px_frac
    rows: List[tuple] = []

    for row in ball.itertuples(index=False):
        frame = int(row.frame)
        group = by_frame.get(frame)
        if group is None or len(group) == 0:
            rows.append((frame, -1, 0, np.nan))
            continue

        use_pitch = not np.isnan(row.pitch_x) and group["pitch_x"].notna().any()
        if use_pitch:
            candidates = group[group["pitch_x"].notna()]
            if candidates.empty:
                rows.append((frame, -1, 0, np.nan))
                continue
            pts = candidates[["pitch_x", "pitch_y"]].to_numpy()
            target = np.array([row.pitch_x, row.pitch_y])
            limit = config.max_distance_m
        else:
            if np.isnan(row.img_x):
                rows.append((frame, -1, 0, np.nan))
                continue
            candidates = group
            pts = candidates[["img_x", "img_y"]].to_numpy()
            target = np.array([row.img_x, row.img_y])
            limit = px_limit

        distances = np.linalg.norm(pts - target, axis=1)
        best = int(np.argmin(distances))
        if distances[best] > limit:
            rows.append((frame, -1, 0, np.nan))
            continue

        track_id = int(candidates.iloc[best]["track_id"])
        rows.append((frame, track_id, int(teams.get(track_id, 0)), float(distances[best])))

    return pd.DataFrame(rows, columns=["frame", "holder", "holder_team", "distance"])


def _smooth_holder(
    holders: Sequence[int],
    config: PossessionConfig,
    teams: Dict[int, int] | None = None,
) -> List[int]:
    """Turn a noisy per-frame nearest-player series into stable possession.

    A new player takes possession only after leading for `confirm_frames`
    consecutive frames; a gap of up to `release_frames` with nobody near the
    ball does not end the current spell, because that gap is exactly what a pass
    in flight looks like.

    **A turnover is a bigger claim than a pass, and needs more evidence.**
    An opponent must lead for `steal_frames` rather than `confirm_frames`. The
    asymmetry is not a tuning knob but a fact about football: in a crowd the
    player nearest the ball alternates between the two teams frame by frame,
    and treating each alternation as a change of possession manufactured 121
    turnovers in six minutes of play where a person tagging it saw about
    twenty. Every spurious one becomes an interception in the event list.

    **And a player who loses the ball to nobody and immediately has it again
    never lost it.** Without `regain_frames` that blink splits one possession
    into two with a fabricated event between them.

    `teams` may be omitted, in which case every change is treated as a pass —
    the behaviour before the asymmetry existed.
    """
    out: List[int] = []
    current = -1
    candidate = -1
    candidate_run = 0
    empty_run = 0
    last_holder = -1

    for holder in holders:
        if holder == -1:
            empty_run += 1
            candidate = -1
            candidate_run = 0
            if empty_run > config.release_frames:
                current = -1
            out.append(current)
            continue

        # The same player, back after a blink with nobody in between.
        if (
            holder == last_holder
            and current == -1
            and empty_run <= config.regain_frames
        ):
            current = holder
            empty_run = candidate_run = 0
            candidate = -1
            out.append(current)
            continue

        empty_run = 0
        if holder == current:
            candidate = -1
            candidate_run = 0
            last_holder = current
            out.append(current)
            continue

        if holder == candidate:
            candidate_run += 1
        else:
            candidate = holder
            candidate_run = 1

        opposed = (
            teams is not None
            and current != -1
            and teams.get(holder, 0) != teams.get(current, 0)
        )
        needed = config.steal_frames if opposed else config.confirm_frames

        # From nobody, the ordinary threshold applies: the ball has to be
        # picked up somehow, and requiring a turnover's evidence to *start*
        # a possession loses most of the real ones.
        if candidate_run >= needed or (current == -1 and candidate_run >= config.confirm_frames):
            current = candidate
            candidate = -1
            candidate_run = 0
        out.append(current)
        if current != -1:
            last_holder = current

    return out


def build_spells(
    holder_frame: pd.DataFrame,
    people: pd.DataFrame,
    ball: pd.DataFrame,
    teams: Dict[int, int],
    classes: Dict[int, int],
    config: PossessionConfig,
) -> List[Spell]:
    """Collapse the smoothed per-frame holder series into possession spells."""
    if holder_frame.empty:
        return []

    df = holder_frame.sort_values("frame").reset_index(drop=True)
    df["smoothed"] = _smooth_holder(df["holder"].tolist(), config, teams)

    times = dict(zip(ball["frame"].astype(int), ball["time_s"]))
    ball_xy = {
        int(r.frame): (np.array([r.pitch_x, r.pitch_y]) if not np.isnan(r.pitch_x) else None)
        for r in ball.itertuples(index=False)
    }
    # Per track, its observed positions in frame order — so a spell can be
    # anchored to the player's first and last *sighting* inside it rather than
    # to two exact frames.
    player_track: Dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for r in people.sort_values("frame").itertuples(index=False):
        if not np.isnan(r.pitch_x):
            player_track[int(r.track_id)].append(
                (int(r.frame), np.array([r.pitch_x, r.pitch_y]))
            )

    def positions_in(track_id: int, f0: int, f1: int) -> tuple[np.ndarray, np.ndarray] | None:
        """The player's first and last known positions within a frame range.

        A spell can outlast the frames the player was actually detected in —
        possession is held through a short gap, and the tracker drops people for
        a frame or two. Demanding a detection at exactly the first and last
        frame of the spell threw away perfectly good possession.
        """
        seen = [(frame, xy) for frame, xy in player_track.get(track_id, ()) if f0 <= frame <= f1]
        if not seen:
            return None
        return seen[0], seen[-1]

    spells: List[Spell] = []
    run_start: int | None = None
    run_holder = -1

    def close(start_idx: int, end_idx: int, track_id: int) -> None:
        if track_id < 0:
            return
        f0 = int(df.loc[start_idx, "frame"])
        f1 = int(df.loc[end_idx, "frame"])
        if (end_idx - start_idx + 1) < config.min_spell_frames:
            return
        anchors = positions_in(track_id, f0, f1)
        if anchors is None:
            # The player was never located on the pitch during this spell, so
            # there is no geometry to build an event from.
            return
        (contact_start, p0), (contact_end, p1) = anchors
        spells.append(
            Spell(
                track_id=track_id,
                team=int(teams.get(track_id, 0)),
                start_frame=f0,
                end_frame=f1,
                start_time=float(times.get(f0, 0.0)),
                end_time=float(times.get(f1, 0.0)),
                start_xy=p0,
                end_xy=p1,
                ball_start_xy=ball_xy.get(contact_start),
                ball_end_xy=ball_xy.get(contact_end),
                is_goalkeeper=classes.get(track_id) == GOALKEEPER,
                contact_start=contact_start,
                contact_end=contact_end,
            )
        )

    for i, holder in enumerate(df["smoothed"].tolist()):
        if holder != run_holder:
            if run_start is not None:
                close(run_start, i - 1, run_holder)
            run_holder = holder
            run_start = i
    if run_start is not None:
        close(run_start, len(df) - 1, run_holder)

    return spells


def possession_share(
    holder_frame: pd.DataFrame,
    teams: Dict[int, int],
    config: PossessionConfig | None = None,
) -> Dict[int, float]:
    """Share of the time each team had the ball, as a percentage.

    Counted frame by frame rather than by summing confirmed spells. Two reasons:
    a spell only exists once possession has been *held* long enough to confirm,
    so summing spells quietly discards every scrappy passage of play; and spells
    need pitch coordinates, which a video the calibration model cannot read does
    not have. Possession is the one number that should survive a failed
    calibration, because it needs no geometry at all — only who was nearest the
    ball, which is answerable in pixels.
    """
    if holder_frame is None or holder_frame.empty:
        return {}

    smoothed = _smooth_holder(
        holder_frame.sort_values("frame")["holder"].tolist(),
        config or PossessionConfig(),
        teams,
    )
    counts: Dict[int, int] = {}
    for holder in smoothed:
        if holder < 0:
            continue
        team = int(teams.get(holder, 0))
        if team:
            counts[team] = counts.get(team, 0) + 1

    total = sum(counts.values())
    if not total:
        return {}
    return {team: round(100.0 * n / total, 1) for team, n in counts.items()}
