"""Re-linking the fragments of one player into a single identity.

ByteTrack decides frame by frame, and it decides in *image* space: two boxes
are the same player if they overlap. That is the right trade for a live feed,
and it is the wrong one here, because the camera moves. When a broadcast camera
pans to follow a break, every box on the screen jumps sideways between frames;
the overlap the tracker is looking for is simply not there, so it retires the
identity and issues a new one to the same person a moment later. Occlusion does
the rest — players stand in front of each other constantly, and each time one
emerges it risks being labelled a stranger.

The result is roughly two and a half identities per player over a clip, which
quietly ruins every statistic that accumulates: distance covered, passes made,
possession held. It also manufactures events, because a possession spell that
ends when an identity ends looks exactly like a turnover.

We are not a live feed. The whole video is on disk, so the association can be
made *offline*, with the answer already known — collect the fragments first,
then decide which belong together while looking at all of them at once. That is
what post-match systems do, and it is strictly more informed than any online
tracker can be.

Two things make it work here that would not be available to the tracker:

  * **Pitch coordinates.** The homography already maps feet to metres on the
    pitch, which cancels the camera motion that broke the tracking in the first
    place. A player who leaps twenty pixels sideways because the camera panned
    has not moved at all in metres. Association in this space is association in
    the world, and the constraint becomes physical: could a human being have
    covered that ground in that time?

  * **Hard temporal exclusion.** A person cannot be in two places at once, so
    two fragments that are ever visible in the same frame are certainly
    different people. That single rule removes the overwhelming majority of
    candidate pairs before any scoring happens, which is why the remaining
    decisions can be made with a fairly simple cost.

Linking runs in rounds of increasing gap, shortest first. A half-second gap
across two metres is close to certain, and resolving it lengthens both
fragments — giving the harder, longer gaps better endpoints and better kit
evidence to be judged on. Deciding the easy cases first and letting them inform
the hard ones is the point; doing every gap at once would throw that away.

Within a round the choice is a global one. Each fragment may hand off to at
most one successor and accept at most one predecessor, which is exactly a
linear assignment problem, so it is solved as one rather than greedily: a pair
that looks good in isolation is correctly refused when both halves are a better
match for somebody else.

The thresholds below were measured, not guessed. `training/scripts/
evaluate_stitch.py` builds ground truth by construction — it cuts long tracks
in half, hands the halves back as strangers, and checks whether they are
reunited — which gives a real precision and recall per gap length. On held-out
splits the current settings rejoin 98% of gaps up to two and a half seconds
with no wrong joins at all.

They also mark where this approach stops. Past roughly three seconds precision
collapses below 0.7: a player can be anywhere within thirty metres by then, and
thirty metres of a football pitch contains most of both teams. Position and kit
colour have no answer there, and neither does any amount of tuning — teammates
are, by design, identical. Closing those gaps needs a signal that identifies a
*person* rather than a location, which in football means the number on his
back. That is why the rounds stop at 3.5 s rather than reaching further and
guessing.

One caveat on the numbers: they come from a single broadcast clip. The
parameters were deliberately taken from the middle of each range that scored
perfectly rather than from its edge, so that footage which behaves a little
differently does not fall off a cliff.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .detect import GOALKEEPER, PLAYER, REFEREE


@dataclass
class StitchConfig:
    """Thresholds for re-linking. Distances are metres on the pitch."""

    enabled: bool = True

    # Gap lengths to attempt, in seconds, easiest first. Each round rebuilds
    # the fragments from the previous round's merges, so a long gap is judged
    # on endpoints that have already been extended by the short-gap round.
    rounds: Sequence[float] = (0.6, 1.6, 3.5)

    # How far a player may have travelled, in two regimes. Sprint speed is
    # 10-11 m/s but nobody holds it, and — more to the point — nobody holds a
    # *direction* while holding it. What the budget needs is not top speed but
    # net displacement, and over several seconds that is far slower than a
    # sprint, because real running curves, checks and doubles back.
    #
    # A single sprint figure applied to a three-second gap authorises thirty
    # metres, which on a pitch means the budget contains half the opposition
    # and stops constraining anything. Splitting it in two keeps short gaps
    # generous, where a genuine burst has to fit, and long gaps tight, where
    # the honest expectation is a much smaller net move. Measured on held-out
    # splits, this beat every single-speed setting at gaps beyond two seconds.
    burst_speed_m_s: float = 9.0
    burst_seconds: float = 0.45
    sustained_speed_m_s: float = 3.0

    # Slack added to every motion budget, absorbing the error in the two
    # endpoint positions themselves. Without it, correct links across short
    # gaps would be rejected: the true displacement there is small enough that
    # measurement noise dominates it entirely.
    #
    # Notably smaller than the 2.7 m mean calibration error, and deliberately.
    # That figure is the *absolute* error of a solved frame, most of which is a
    # smooth bias shared by everything in the same region of the same shot. Two
    # feet half a second apart inherit almost the same bias, so what matters
    # here is the much smaller relative error — and sizing this by the absolute
    # figure would open the budget wide enough to swallow the neighbouring
    # player, who is usually about three metres away.
    position_tolerance_m: float = 1.8

    # Velocity is extrapolated across the gap, but only so far: players change
    # direction, and a stale velocity held for a second points confidently at
    # the wrong place. Past this horizon the prediction simply stops moving.
    velocity_horizon_s: float = 0.45
    velocity_window: int = 5

    # Kit colour, as the 15-D descriptor the team classifier already computes
    # per track. It is a supporting signal, not a deciding one: teammates share
    # a kit by definition, so this separates a player from an opponent or an
    # official, never from his own team-mate. Only a gross mismatch is refused.
    kit_weight: float = 0.30
    max_kit_distance: float = 0.55

    # How much better than the runner-up the winning link must be. On a pitch
    # the nearest player is routinely three or four metres away, so a fragment
    # that reappears has several physically plausible owners and the cheapest
    # of them is frequently not the right one. Requiring a clear margin turns
    # those coin-flips into refusals — which is the correct answer, because a
    # wrong join is far more damaging than a missed one: it hands one player's
    # work to another, while a miss only leaves the fragmentation we started
    # with. Set to 1.0 to accept every winner regardless of how close the race.
    #
    # Tuned together with the budget above rather than separately: once the
    # budget stopped authorising thirty-metre moves, most races were no longer
    # close, and the margin could be relaxed from a value that had been
    # compensating for a budget that was too wide.
    max_ambiguity: float = 0.75

    # Two fragments assigned to different teams are different people. This is
    # reliable — the classifier votes over a whole track and reports ~0.98
    # confidence — but only when both fragments actually have a team.
    require_same_team: bool = True
    # A fragment shorter than this has too few observations to place or colour
    # with any confidence; it is left out of the linking rather than guessed at.
    min_tracklet_obs: int = 3

    # Fallback for footage the calibrator could not solve, where there are no
    # metres to reason in. Image-space linking is far weaker — camera motion is
    # exactly what it cannot see through — so it is deliberately conservative:
    # a fraction of the frame width, and no velocity extrapolation.
    image_tolerance_fraction: float = 0.06
    allow_image_fallback: bool = True


@dataclass
class Tracklet:
    """One uninterrupted fragment of one player, reduced to its endpoints."""

    track_id: int
    start_frame: int
    end_frame: int
    n_obs: int
    cls: int
    team: int | None
    # Pitch metres, or None where the clip was never calibrated here.
    head: np.ndarray | None
    tail: np.ndarray | None
    head_velocity: np.ndarray
    tail_velocity: np.ndarray
    # Image pixels, always present — the fallback space.
    head_img: np.ndarray
    tail_img: np.ndarray
    kit: np.ndarray | None

    @property
    def has_pitch(self) -> bool:
        return self.head is not None and self.tail is not None


@dataclass
class StitchReport:
    """What the linking did, for the diagnostics block and for tests."""

    tracks_before: int = 0
    tracks_after: int = 0
    merges: int = 0
    per_round: List[dict] = field(default_factory=list)
    used_image_fallback: bool = False

    @property
    def reduction(self) -> float:
        if not self.tracks_before:
            return 0.0
        return 1.0 - (self.tracks_after / self.tracks_before)

    def as_dict(self) -> dict:
        return {
            "tracks_before": self.tracks_before,
            "tracks_after": self.tracks_after,
            "merges": self.merges,
            "reduction": round(self.reduction, 3),
            "rounds": self.per_round,
            "image_fallback": self.used_image_fallback,
        }


# --------------------------------------------------------------- tracklets


def _endpoint(values: np.ndarray, window: int, at_start: bool) -> np.ndarray | None:
    """A robust position for one end of a fragment.

    The median of the first (or last) few observations, not the single outermost
    one: the frame where a track appears or dies is frequently the frame where
    the detection was worst, which is why it appeared or died there.
    """
    if len(values) == 0:
        return None
    slice_ = values[:window] if at_start else values[-window:]
    slice_ = slice_[~np.isnan(slice_).any(axis=1)]
    if len(slice_) == 0:
        return None
    return np.median(slice_, axis=0).astype(np.float32)


def _velocity(
    positions: np.ndarray, frames: np.ndarray, window: int, at_start: bool
) -> np.ndarray:
    """Metres (or pixels) per frame over the last few observations of an end.

    A straight first-to-last difference across the window rather than a fit:
    the window is short, and a fit adds nothing except sensitivity to the
    single bad frame a fit is supposed to survive.
    """
    if len(positions) < 2:
        return np.zeros(2, dtype=np.float32)
    idx = slice(0, window) if at_start else slice(max(len(positions) - window, 0), None)
    pos, frm = positions[idx], frames[idx]
    keep = ~np.isnan(pos).any(axis=1)
    pos, frm = pos[keep], frm[keep]
    if len(pos) < 2:
        return np.zeros(2, dtype=np.float32)
    span = float(frm[-1] - frm[0])
    if span <= 0:
        return np.zeros(2, dtype=np.float32)
    return ((pos[-1] - pos[0]) / span).astype(np.float32)


def build_tracklets(
    people: pd.DataFrame,
    teams: Dict[int, int],
    classes: Dict[int, int],
    kits: Dict[int, np.ndarray] | None,
    config: StitchConfig,
) -> List[Tracklet]:
    """Reduce every track in `people` to the endpoints the linking needs."""
    if people.empty:
        return []

    kits = kits or {}
    out: List[Tracklet] = []
    ordered = people.sort_values(["track_id", "frame"])

    for track_id, group in ordered.groupby("track_id", sort=True):
        if len(group) < config.min_tracklet_obs:
            continue
        frames = group["frame"].to_numpy()
        pitch = group[["pitch_x", "pitch_y"]].to_numpy(dtype=np.float32)
        image = group[["img_x", "img_y"]].to_numpy(dtype=np.float32)
        window = config.velocity_window

        head_img = _endpoint(image, window, at_start=True)
        tail_img = _endpoint(image, window, at_start=False)
        if head_img is None or tail_img is None:
            continue

        out.append(
            Tracklet(
                track_id=int(track_id),
                start_frame=int(frames[0]),
                end_frame=int(frames[-1]),
                n_obs=len(group),
                cls=int(classes.get(int(track_id), PLAYER)),
                team=teams.get(int(track_id)),
                head=_endpoint(pitch, window, at_start=True),
                tail=_endpoint(pitch, window, at_start=False),
                head_velocity=_velocity(pitch, frames, window, at_start=True),
                tail_velocity=_velocity(pitch, frames, window, at_start=False),
                head_img=head_img,
                tail_img=tail_img,
                kit=kits.get(int(track_id)),
            )
        )
    return out


# ------------------------------------------------------------------ scoring


def _kit_distance(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a, np.float32) - np.asarray(b, np.float32)))


def _predict(
    position: np.ndarray, velocity: np.ndarray, frames: float, horizon_frames: float
) -> np.ndarray:
    """Where a fragment's end would be `frames` later, holding its velocity.

    The extrapolation is capped at the horizon: beyond that the prediction
    freezes rather than continuing to fly off in a direction the player has
    almost certainly abandoned.
    """
    return position + velocity * min(float(frames), float(horizon_frames))


def _reach(gap_s: float, config: StitchConfig) -> float:
    """How far a player could plausibly have moved in `gap_s` seconds."""
    burst = min(gap_s, config.burst_seconds) * config.burst_speed_m_s
    rest = max(0.0, gap_s - config.burst_seconds) * config.sustained_speed_m_s
    return burst + rest


def _pair_cost(
    a: Tracklet, b: Tracklet, fps: float, frame_width: float, config: StitchConfig
) -> tuple[float | None, bool]:
    """Cost of declaring `a` and `b` the same player, or None if impossible.

    Returns `(cost, used_image_space)`. Every rejection here is a statement
    about the world — the two were seen together, they are different kinds of
    person, they wear different kits, or no human could have covered the ground.
    """
    # A person is in one place at a time. This is the only rule that needs no
    # threshold, and it eliminates most candidate pairs outright.
    gap_frames = b.start_frame - a.end_frame
    if gap_frames <= 0:
        return None, False

    # Referees are not players and keepers are not outfielders; a detector that
    # confuses them on single frames has already been out-voted by now.
    if a.cls != b.cls:
        return None, False

    if config.require_same_team and a.team is not None and b.team is not None:
        if a.team != b.team:
            return None, False

    kit_distance = _kit_distance(a.kit, b.kit)
    if kit_distance is not None and kit_distance > config.max_kit_distance:
        return None, False

    gap_s = gap_frames / fps if fps > 0 else 0.0
    horizon_frames = config.velocity_horizon_s * fps

    if a.has_pitch and b.has_pitch:
        # Predict from both ends and average. A one-sided prediction inherits
        # whichever endpoint happened to be noisier; meeting in the middle does
        # not, and it costs nothing.
        forward = _predict(a.tail, a.tail_velocity, gap_frames, horizon_frames)
        backward = _predict(b.head, -b.head_velocity, gap_frames, horizon_frames)
        distance = 0.5 * (
            float(np.linalg.norm(forward - b.head))
            + float(np.linalg.norm(backward - a.tail))
        )
        budget = _reach(gap_s, config) + config.position_tolerance_m
        used_image = False
    elif config.allow_image_fallback:
        # No homography here. Fall back to pixels, with no velocity term: in
        # image space the dominant motion is the camera's, and extrapolating it
        # as though it were the player's is worse than not extrapolating.
        distance = float(np.linalg.norm(b.head_img - a.tail_img))
        budget = config.image_tolerance_fraction * frame_width * (1.0 + gap_s)
        used_image = True
    else:
        return None, False

    if budget <= 0 or distance > budget:
        return None, False

    cost = distance / budget
    if kit_distance is not None:
        cost += config.kit_weight * (kit_distance / config.max_kit_distance)
    return float(cost), used_image


# ------------------------------------------------------------------ linking


class _Union:
    """Union-find over track ids, with the earliest-starting id as the root."""

    def __init__(self, ids: Iterable[int]):
        self._parent = {int(i): int(i) for i in ids}

    def find(self, i: int) -> int:
        root = int(i)
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression, so a long chain does not cost more each lookup.
        node = int(i)
        while self._parent[node] != root:
            self._parent[node], node = root, self._parent[node]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Keep the smaller id as the root purely so the surviving ids are
            # stable and reproducible between runs.
            root, other = (ra, rb) if ra < rb else (rb, ra)
            self._parent[other] = root

    def mapping(self) -> Dict[int, int]:
        return {i: self.find(i) for i in self._parent}


def _link_round(
    tracklets: Sequence[Tracklet],
    max_gap_frames: float,
    fps: float,
    frame_width: float,
    config: StitchConfig,
) -> tuple[List[tuple[int, int]], bool]:
    """One global assignment of fragment-ends to fragment-starts."""
    n = len(tracklets)
    if n < 2:
        return [], False

    LARGE = 1e6
    cost = np.full((n, n), LARGE, dtype=np.float64)
    feasible = np.zeros((n, n), dtype=bool)
    used_image = False

    for i, a in enumerate(tracklets):
        for j, b in enumerate(tracklets):
            if i == j or (b.start_frame - a.end_frame) > max_gap_frames:
                continue
            value, image_space = _pair_cost(a, b, fps, frame_width, config)
            if value is None:
                continue
            cost[i, j] = value
            feasible[i, j] = True
            used_image = used_image or image_space

    if not feasible.any():
        return [], False

    rows, cols = linear_sum_assignment(cost)

    pairs = []
    for i, j in zip(rows, cols):
        if not feasible[i, j]:
            continue
        if not _decisive(cost, feasible, i, j, config.max_ambiguity):
            continue
        pairs.append((tracklets[i].track_id, tracklets[j].track_id))
    return pairs, used_image


def _decisive(
    cost: np.ndarray, feasible: np.ndarray, i: int, j: int, ratio: float
) -> bool:
    """Is `i -> j` clearly the best link for both of its ends?

    Compared against the runner-up along the row (other fragments this end
    could hand off to) and along the column (other fragments that could hand
    off to this one). A win by a hair in either direction is not a win: it
    means the evidence does not actually distinguish two players, and the pair
    is dropped rather than guessed.
    """
    if ratio >= 1.0:
        return True

    def runner_up(values: np.ndarray, mask: np.ndarray, skip: int) -> float:
        others = values[mask]
        if len(others) <= 1:
            return np.inf
        candidates = np.delete(values, skip)[np.delete(mask, skip)]
        return float(candidates.min()) if len(candidates) else np.inf

    best = float(cost[i, j])
    if best <= 0:
        return True
    second = min(
        runner_up(cost[i], feasible[i], j),
        runner_up(cost[:, j], feasible[:, j], i),
    )
    return second == np.inf or best <= ratio * second


def _merge_tracklets(members: Sequence[Tracklet], config: StitchConfig) -> Tracklet:
    """Fold a chain of fragments into one, for the next round to work on."""
    chain = sorted(members, key=lambda t: t.start_frame)
    first, last = chain[0], chain[-1]

    kits = [t.kit for t in chain if t.kit is not None]
    weights = [t.n_obs for t in chain if t.kit is not None]
    kit = (
        np.average(np.stack(kits), axis=0, weights=weights).astype(np.float32)
        if kits
        else None
    )

    teams = [t.team for t in chain if t.team is not None]
    return Tracklet(
        track_id=min(t.track_id for t in chain),
        start_frame=first.start_frame,
        end_frame=last.end_frame,
        n_obs=sum(t.n_obs for t in chain),
        cls=max(set(t.cls for t in chain), key=[t.cls for t in chain].count),
        team=max(set(teams), key=teams.count) if teams else None,
        head=first.head,
        tail=last.tail,
        head_velocity=first.head_velocity,
        tail_velocity=last.tail_velocity,
        head_img=first.head_img,
        tail_img=last.tail_img,
        kit=kit,
    )


def link(
    tracklets: Sequence[Tracklet],
    fps: float,
    frame_width: float,
    config: StitchConfig | None = None,
) -> tuple[Dict[int, int], StitchReport]:
    """Decide which fragments are the same player.

    Returns a mapping from every original track id to the id it should become,
    plus a report of what happened.
    """
    config = config or StitchConfig()
    report = StitchReport(tracks_before=len(tracklets), tracks_after=len(tracklets))
    if not config.enabled or len(tracklets) < 2:
        return {t.track_id: t.track_id for t in tracklets}, report

    union = _Union(t.track_id for t in tracklets)
    working = list(tracklets)

    for max_gap_s in config.rounds:
        pairs, used_image = _link_round(
            working, max_gap_s * fps, fps, frame_width, config
        )
        report.used_image_fallback = report.used_image_fallback or used_image
        report.per_round.append(
            {"max_gap_s": float(max_gap_s), "merges": len(pairs)}
        )
        if not pairs:
            continue

        for a, b in pairs:
            union.union(a, b)

        # Rebuild the fragments so the next, longer round sees the chains this
        # round produced rather than the pieces it started from.
        by_root: Dict[int, List[Tracklet]] = {}
        current = union.mapping()
        for tracklet in working:
            by_root.setdefault(current[tracklet.track_id], []).append(tracklet)
        working = [
            group[0] if len(group) == 1 else _merge_tracklets(group, config)
            for group in by_root.values()
        ]

    mapping = union.mapping()
    report.tracks_after = len(set(mapping.values()))
    report.merges = report.tracks_before - report.tracks_after
    return mapping, report


# ------------------------------------------------------------------- apply


def apply_mapping(people: pd.DataFrame, mapping: Dict[int, int]) -> pd.DataFrame:
    """Rewrite track ids in an observation frame.

    Fragments that were never considered for linking keep the id they had, so
    a partial mapping is safe to pass.
    """
    if people.empty or not mapping:
        return people
    out = people.copy()
    out["track_id"] = out["track_id"].map(lambda t: mapping.get(int(t), int(t))).astype(int)
    return out.sort_values(["track_id", "frame"]).reset_index(drop=True)


def remap_dict(values: Dict[int, int], mapping: Dict[int, int]) -> Dict[int, int]:
    """Fold a per-track lookup onto the surviving ids by majority."""
    if not mapping:
        return values
    votes: Dict[int, Dict[int, int]] = {}
    for track_id, value in values.items():
        root = mapping.get(int(track_id), int(track_id))
        votes.setdefault(root, {})
        votes[root][value] = votes[root].get(value, 0) + 1
    return {root: max(v, key=v.get) for root, v in votes.items()}


def stitch(
    people: pd.DataFrame,
    teams: Dict[int, int],
    classes: Dict[int, int],
    kits: Dict[int, np.ndarray] | None,
    fps: float,
    frame_width: float,
    config: StitchConfig | None = None,
) -> tuple[pd.DataFrame, Dict[int, int], StitchReport]:
    """Build fragments, link them, and return the re-identified observations."""
    config = config or StitchConfig()
    if not config.enabled or people.empty:
        return people, {}, StitchReport()

    tracklets = build_tracklets(people, teams, classes, kits, config)
    mapping, report = link(tracklets, fps, frame_width, config)
    if not report.merges:
        return people, mapping, report
    return apply_mapping(people, mapping), mapping, report
