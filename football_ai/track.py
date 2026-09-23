"""Tracking: giving every player a stable identity across frames, and keeping
the ball's path continuous through the frames where it was not seen.

Detection alone answers "who is on the pitch"; tracking answers "is this the
same person as a moment ago", which is the prerequisite for every statistic
that accumulates — distance run, passes made, possession held.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import supervision as sv

from .detect import GOALKEEPER, PLAYER, REFEREE


@dataclass
class TrackConfig:
    """ByteTrack settings, chosen for a crowded pitch analysed at ~12 fps.

    ByteTrack works in two passes: confident detections may start a new track,
    and *weak* ones are then used to keep existing tracks alive. The settings
    below lean into that split. A high activation threshold stops noise from
    spawning identities, while a low detection confidence upstream keeps plenty
    of weak boxes available to carry a player through a moment of occlusion —
    which on a football pitch happens constantly, players being the things that
    stand in front of each other.
    """

    # Confidence a detection needs before it may start a *new* identity.
    track_activation_threshold: float = 0.40
    # How long an unseen track survives, in frames — scaled internally by the
    # frame rate. Nearly three seconds at the rate we analyse: long enough to
    # ride out a player disappearing behind a crowd near the touchline.
    lost_track_buffer: int = 90
    minimum_matching_threshold: float = 0.85
    frame_rate: int = 25
    # A track seen only a handful of times is a detection glitch, not a player.
    min_track_length: int = 8


def make_tracker(config: TrackConfig, frame_rate: float) -> sv.ByteTrack:
    return sv.ByteTrack(
        track_activation_threshold=config.track_activation_threshold,
        lost_track_buffer=config.lost_track_buffer,
        minimum_matching_threshold=config.minimum_matching_threshold,
        frame_rate=int(round(frame_rate)) or config.frame_rate,
    )


def foot_position(bbox: Sequence[float]) -> np.ndarray:
    """The point on the ground a player is standing on.

    The bottom-centre of the box, not its centre: only the feet are actually on
    the pitch plane, and the homography maps that plane. Using the box centre
    would place every player several metres further from the camera than they
    are.
    """
    x1, _, x2, y2 = bbox
    return np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)


class TrackStore:
    """Flat, append-only record of every observation, materialised as a frame.

    Kept as plain lists during processing and converted to a DataFrame once at
    the end — appending to a DataFrame row by row is quadratic and would
    dominate the runtime of a long clip.
    """

    COLUMNS = (
        "frame", "time_s", "track_id", "cls",
        "x1", "y1", "x2", "y2",
        "img_x", "img_y", "pitch_x", "pitch_y",
    )

    def __init__(self) -> None:
        self._rows: List[tuple] = []
        self._ball: List[tuple] = []

    def add_people(
        self,
        frame: int,
        time_s: float,
        detections: sv.Detections,
        pitch_xy: np.ndarray | None,
    ) -> None:
        if len(detections) == 0 or detections.tracker_id is None:
            return
        for i in range(len(detections)):
            track_id = detections.tracker_id[i]
            if track_id is None:
                continue
            bbox = detections.xyxy[i]
            foot = foot_position(bbox)
            px, py = (pitch_xy[i] if pitch_xy is not None else (np.nan, np.nan))
            self._rows.append(
                (
                    frame, time_s, int(track_id), int(detections.class_id[i]),
                    float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]),
                    float(foot[0]), float(foot[1]), float(px), float(py),
                )
            )

    def add_ball(
        self,
        frame: int,
        time_s: float,
        centre: np.ndarray | None,
        pitch_xy: np.ndarray | None,
        confidence: float = 0.0,
    ) -> None:
        if centre is None:
            self._ball.append((frame, time_s, np.nan, np.nan, np.nan, np.nan, 0.0))
            return
        px, py = (pitch_xy if pitch_xy is not None else (np.nan, np.nan))
        self._ball.append(
            (frame, time_s, float(centre[0]), float(centre[1]), float(px), float(py), float(confidence))
        )

    # ---- materialisation ----------------------------------------------------

    def people_frame(self, min_track_length: int = 0) -> pd.DataFrame:
        df = pd.DataFrame(self._rows, columns=list(self.COLUMNS))
        if df.empty:
            return df
        return drop_short_tracks(df, min_track_length)

    def ball_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._ball,
            columns=["frame", "time_s", "img_x", "img_y", "pitch_x", "pitch_y", "conf"],
        )


def drop_short_tracks(people: pd.DataFrame, min_length: int) -> pd.DataFrame:
    """Discard identities with too few observations to be a real person.

    Applied *after* re-linking rather than before it: a two-frame fragment is
    noise on its own, but the same two frames joined to the fifty either side
    of them are the middle of a player's run. Filtering first would throw away
    the very pieces the linking exists to reclaim.
    """
    if people.empty or min_length <= 0:
        return people.reset_index(drop=True)
    counts = people.groupby("track_id")["frame"].transform("size")
    return people[counts >= min_length].reset_index(drop=True)


def interpolate_ball(ball: pd.DataFrame, max_gap: int = 25) -> pd.DataFrame:
    """Fill short gaps in the ball's path, and smooth the result.

    The ball disappears constantly — occluded by a player, lost in the crowd
    behind, motion-blurred to nothing during a hard pass. Bridging those gaps is
    what lets a pass be measured end to end. Gaps longer than `max_gap` frames
    are left empty on purpose: over a second without sight of the ball, a
    straight line between the two ends is a guess, not an observation.
    """
    if ball.empty:
        return ball

    out = ball.copy().sort_values("frame").reset_index(drop=True)
    seen = out["img_x"].notna()
    if not seen.any():
        return out

    # Identify runs of missing rows and only interpolate the short ones.
    missing = ~seen
    group = (missing != missing.shift()).cumsum()
    long_gap = pd.Series(False, index=out.index)
    for _, idx in out[missing].groupby(group[missing]).groups.items():
        idx = pd.Index(idx)
        # A gap before the first or after the last sighting has nothing to
        # interpolate between; leave it.
        if len(idx) > max_gap or idx.min() == 0 or idx.max() == len(out) - 1:
            long_gap.loc[idx] = True

    for col in ("img_x", "img_y", "pitch_x", "pitch_y"):
        filled = out[col].interpolate(method="linear", limit_direction="both")
        # A 3-frame rolling median knocks out the single-frame jumps a
        # false-positive detection causes without blunting a real fast pass.
        smoothed = filled.rolling(window=3, center=True, min_periods=1).median()
        # Blank the long gaps last: smoothing would otherwise bleed a value one
        # frame into each end of a gap we deliberately refused to fill.
        smoothed[long_gap] = np.nan
        out[col] = smoothed

    out["interpolated"] = missing & ~long_gap
    return out


def track_classes(people: pd.DataFrame) -> Dict[int, int]:
    """The settled class of each track: player, goalkeeper or referee.

    A detector will occasionally call a goalkeeper a player, or vice versa, on
    individual frames. The class a track holds most often is the right one.
    """
    if people.empty:
        return {}
    modes = people.groupby("track_id")["cls"].agg(lambda s: s.value_counts().idxmax())
    return {int(k): int(v) for k, v in modes.items()}


def smooth_positions(people: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Rolling-median smoothing of each track's pitch position.

    Detection boxes jitter by a few pixels frame to frame, which the homography
    magnifies into tens of centimetres of phantom movement — enough, summed over
    90 minutes, to add kilometres to every player's distance covered.
    """
    if people.empty:
        return people
    out = people.sort_values(["track_id", "frame"]).copy()
    for col in ("pitch_x", "pitch_y"):
        out[col] = (
            out.groupby("track_id")[col]
            .transform(lambda s: s.rolling(window, center=True, min_periods=1).median())
        )
    return out.reset_index(drop=True)
