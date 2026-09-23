"""Finding the cuts, so a broadcast can be analysed instead of only a clip.

Everything upstream of here assumes the camera moves continuously: the tracker
matches a player to the box nearest where they were, the calibration carries a
homography forward through frames it could not solve, the ball search looks
near its last sighting. Those assumptions hold beautifully for a single
unbroken shot and collapse at the first edit.

A televised match is nothing but edits. Between kick-off and the whistle there
are hundreds of them — replays, close-ups on a face, the bench, the crowd, the
camera behind the goal — and each one breaks the same three things:

  * **Tracking.** The frame after a cut has no spatial relationship to the one
    before. Every box has moved, so every identity is retired and reissued, and
    a player accumulates a fresh identity per cut.
  * **Calibration.** A homography solved on the wide camera is meaningless on
    the tight one, and carrying it across places players metres from where they
    stand — or off the pitch entirely.
  * **Events.** A replay shows the same goal a second time. Nothing downstream
    can tell it apart from the goal, so it is counted twice.

The third is the one that would embarrass you in front of a customer, and none
of the three can be fixed by better models. They need the video split into
shots first, which is what this does.

**How a cut is recognised.** Consecutive frames of one shot look alike, however
fast the camera moves; frames either side of a cut usually do not. So each
frame is reduced to a coarse greyscale grid — where the light and dark parts of
the picture are — and compared with the one before it. A large jump is a cut.

The obvious signal, a colour histogram, was tried first and measured worse.
Between the faintest real cut and the loudest ordinary frame it left a margin
of 1.5x, against 3.3x for the grid — and at that spacing the histogram could
not be thresholded without either missing cuts or inventing them. The reason is the hardest and most common case in football. A cut
from one wide shot of the pitch to another wide shot of the same pitch barely
changes the colours at all — it is the same grass, the same two kits, the same
crowd — but it rearranges completely where everything sits. Layout is what an
edit destroys; colour is what it preserves.

The threshold is relative, not absolute. A fixed one cannot work: a locked-off
tactical camera and a hand-held broadcast feed following a counter-attack sit
at completely different baselines, and any constant that catches the cuts in
the first will fire constantly on the second. What matters is not how much the
picture changed but how much it changed *compared to how much it has been
changing* — so the threshold is a multiple of the recent median distance,
which adapts to the footage as it plays.

**What a shot is worth.** Splitting is only half the job. A replay is a shot,
and so is a close-up of a manager, and neither is play. They are told apart by
whether the pitch can be located in them: a wide shot of the field solves a
homography, a face does not. A shot that never calibrates is not football, and
saying so keeps the crowd out of the possession numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np


@dataclass
class SegmentConfig:
    """Thresholds for splitting a video into shots."""

    enabled: bool = True

    # Side of the greyscale grid each frame is reduced to. Coarse on purpose:
    # at 16 the grass texture sliding past during a pan averages away, while
    # the arrangement of players, stands and shadow that an edit rearranges
    # survives. Finer grids turn ordinary camera movement into cuts.
    grid: int = 16

    # A cut is a distance this many times the recent median. Relative rather
    # than absolute, because a tripod and a hand-held broadcast camera differ
    # by more than any constant could span.
    cut_ratio: float = 4.0
    # Below this, nothing is a cut however quiet the footage has been. Without
    # a floor, a perfectly still shot drives the median towards zero and then
    # every twitch clears the ratio. Measured on constructed cuts, ordinary
    # play peaks at 0.014 and the faintest cut sits at 0.045, so this sits
    # between them with room on both sides.
    min_cut_distance: float = 0.022
    # How many recent frames the median is taken over — a few seconds at the
    # rate we analyse, long enough to describe the current shot and short
    # enough to follow the film into the next one.
    window: int = 40

    # A shot shorter than this is a flash: a transition wipe, a graphic, a
    # single misread frame. It is recorded but never treated as play.
    min_shot_frames: int = 6
    # A shot is play only if the pitch was found in at least this share of the
    # frames tried. Replays and close-ups fail it; wide match footage passes it
    # comfortably.
    min_calibrated_share: float = 0.35


@dataclass
class Shot:
    """One continuous run of frames between two cuts."""

    index: int
    start_frame: int
    end_frame: int
    frames: int = 0
    calibrated: int = 0
    attempted: int = 0

    @property
    def calibrated_share(self) -> float:
        return self.calibrated / self.attempted if self.attempted else 0.0

    def is_play(self, config: SegmentConfig) -> bool:
        """Is this shot wide footage of the match, rather than a replay or a face?"""
        if self.frames < config.min_shot_frames:
            return False
        if not self.attempted:
            # Nothing was tried here — no evidence either way, so do not
            # discard it. Downstream stages simply have no metres to work in.
            return True
        return self.calibrated_share >= config.min_calibrated_share

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frames": self.frames,
            "calibrated_share": round(self.calibrated_share, 3),
        }


def signature(frame: np.ndarray, config: SegmentConfig) -> np.ndarray:
    """One frame as a coarse greyscale grid, values in [0, 1]."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(
        grey, (config.grid, config.grid), interpolation=cv2.INTER_AREA
    )
    return small.astype(np.float32) / 255.0


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """How unlike two frame signatures are, in [0, 1].

    Mean absolute difference cell by cell: how much of the picture's layout
    changed. A pan slides content between neighbouring cells and moves this a
    little; an edit replaces every cell at once and moves it a lot.
    """
    return float(np.abs(np.asarray(a) - np.asarray(b)).mean())


class ShotDetector:
    """Splits a stream of frames into shots, one frame at a time.

    Fed frames in order; `observe` returns True when the frame it was given
    begins a new shot. The caller uses that to reset whatever it is carrying
    forward — the tracker's identities, the ball's last position, the
    homography — because none of it survives an edit.
    """

    def __init__(self, config: SegmentConfig | None = None):
        self.config = config or SegmentConfig()
        self._previous: np.ndarray | None = None
        self._recent: List[float] = []
        self._shots: List[Shot] = []
        self._current: Shot | None = None

    # ---- driving ------------------------------------------------------------

    def observe(self, frame_index: int, frame: np.ndarray) -> bool:
        """Record a frame. Returns True if it starts a new shot."""
        current = signature(frame, self.config)
        cut = False

        if self._previous is None:
            self._begin(frame_index)
        else:
            d = distance(self._previous, current)
            if self._is_cut(d):
                self._begin(frame_index)
                cut = True
                # The distance that made the cut belongs to neither shot;
                # keeping it would poison the new shot's baseline immediately.
                self._recent.clear()
            else:
                self._recent.append(d)
                if len(self._recent) > self.config.window:
                    self._recent.pop(0)

        self._previous = current
        if self._current is not None:
            self._current.end_frame = frame_index
            self._current.frames += 1
        return cut

    def _is_cut(self, d: float) -> bool:
        if not self.config.enabled:
            return False
        if d < self.config.min_cut_distance:
            return False
        if not self._recent:
            # First comparison of a shot: only a very large jump counts, since
            # there is no baseline yet to be a multiple of.
            return d >= self.config.min_cut_distance * 2.0
        baseline = float(np.median(self._recent))
        return d >= max(
            self.config.min_cut_distance, baseline * self.config.cut_ratio
        )

    def _begin(self, frame_index: int) -> None:
        self._current = Shot(
            index=len(self._shots), start_frame=frame_index, end_frame=frame_index
        )
        self._shots.append(self._current)

    def note_calibration(self, solved: bool) -> None:
        """Record whether the pitch was found in the frame just observed."""
        if self._current is None:
            return
        self._current.attempted += 1
        self._current.calibrated += int(bool(solved))

    # ---- reading ------------------------------------------------------------

    @property
    def shots(self) -> List[Shot]:
        return list(self._shots)

    def playable(self) -> List[Shot]:
        return [s for s in self._shots if s.is_play(self.config)]

    def shot_of(self, frame_index: int) -> Shot | None:
        for shot in self._shots:
            if shot.start_frame <= frame_index <= shot.end_frame:
                return shot
        return None

    def frame_to_shot(self) -> Dict[int, int]:
        """`{frame index: shot index}` for every frame in every shot's span."""
        out: Dict[int, int] = {}
        for shot in self._shots:
            out[shot.start_frame] = shot.index
            out[shot.end_frame] = shot.index
        return out

    def summary(self) -> dict:
        play = self.playable()
        played = sum(s.frames for s in play)
        total = sum(s.frames for s in self._shots)
        return {
            "shots": len(self._shots),
            "cuts": max(len(self._shots) - 1, 0),
            "playable_shots": len(play),
            "playable_frames": played,
            "playable_share": round(played / total, 3) if total else 0.0,
            "longest_shot_frames": max((s.frames for s in self._shots), default=0),
        }


def split_by_shot(
    frames: Sequence[int], shots: Sequence[Shot]
) -> Dict[int, int]:
    """Label each frame index with the shot it belongs to.

    Frames that fall in no shot — which should not happen, but can if a caller
    skips frames the detector never saw — are left out rather than guessed at.
    """
    label: Dict[int, int] = {}
    if not shots:
        return label
    bounds = [(s.start_frame, s.end_frame, s.index) for s in shots]
    for f in frames:
        for start, end, index in bounds:
            if start <= f <= end:
                label[int(f)] = index
                break
    return label
