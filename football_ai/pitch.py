"""Pitch geometry: the single source of truth for "where on the field is this?".

Everything downstream — events, statistics, the radar, the tagger's canvas —
agrees only because they all convert through the definitions in this module.

Three coordinate spaces are in play:

  image space   pixels in the video frame; origin top-left
  pitch space   metres on a real pitch, x along the length (0..105 by default),
                y across the width (0..68). This is where football happens.
  tagger space  the 120 x 80 grid the manual tagger has always used, matching
                the StatsBomb convention. Only the API boundary speaks it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

# The tagger's canvas has been 120 x 80 since the first version of the app, and
# every previously saved match is stored in it. It stays the wire format.
TAGGER_LENGTH = 120.0
TAGGER_WIDTH = 80.0


@dataclass(frozen=True)
class PitchConfig:
    """Dimensions of the pitch, in centimetres, plus the 32 landmarks a model
    can actually see on it.

    The defaults match the pitch the keypoint dataset was annotated against, so
    landmark *i* returned by the pose model is `vertices[i]` here. Do not
    reorder `vertices` — the mapping to model output is positional.
    """

    # FIFA standard, in centimetres. The keypoint model was annotated against
    # real pitch landmarks, so feeding the homography the *real* dimensions is
    # what makes the resulting metres actually metres. (The upstream reference
    # config draws a 120 x 70 pitch with a 20.15 m penalty area; that is a
    # drawing convenience, not a pitch, and it would stretch every distance we
    # measure by about 14%.)
    length: int = 10500
    width: int = 6800
    penalty_box_length: int = 1650
    penalty_box_width: int = 4032
    goal_box_length: int = 550
    goal_box_width: int = 1832
    centre_circle_radius: int = 915
    penalty_spot_distance: int = 1100
    goal_width: int = 732

    @property
    def length_m(self) -> float:
        return self.length / 100.0

    @property
    def width_m(self) -> float:
        return self.width / 100.0

    @property
    def vertices(self) -> List[Tuple[float, float]]:
        """The 32 landmarks in centimetres, in model-output order."""
        L, W = self.length, self.width
        pb_l, pb_w = self.penalty_box_length, self.penalty_box_width
        gb_l, gb_w = self.goal_box_length, self.goal_box_width
        r, spot = self.centre_circle_radius, self.penalty_spot_distance
        return [
            (0, 0),                                  # 0  left goal-line, top touchline
            (0, (W - pb_w) / 2),                     # 1  left penalty box, top
            (0, (W - gb_w) / 2),                     # 2  left goal box, top
            (0, (W + gb_w) / 2),                     # 3  left goal box, bottom
            (0, (W + pb_w) / 2),                     # 4  left penalty box, bottom
            (0, W),                                  # 5  left goal-line, bottom touchline
            (gb_l, (W - gb_w) / 2),                  # 6
            (gb_l, (W + gb_w) / 2),                  # 7
            (spot, W / 2),                           # 8  left penalty spot
            (pb_l, (W - pb_w) / 2),                  # 9
            (pb_l, (W - gb_w) / 2),                  # 10
            (pb_l, (W + gb_w) / 2),                  # 11
            (pb_l, (W + pb_w) / 2),                  # 12
            (L / 2, 0),                              # 13 halfway, top touchline
            (L / 2, W / 2 - r),                      # 14 centre circle, top
            (L / 2, W / 2 + r),                      # 15 centre circle, bottom
            (L / 2, W),                              # 16 halfway, bottom touchline
            (L - pb_l, (W - pb_w) / 2),              # 17
            (L - pb_l, (W - gb_w) / 2),              # 18
            (L - pb_l, (W + gb_w) / 2),              # 19
            (L - pb_l, (W + pb_w) / 2),              # 20
            (L - spot, W / 2),                       # 21 right penalty spot
            (L - gb_l, (W - gb_w) / 2),              # 22
            (L - gb_l, (W + gb_w) / 2),              # 23
            (L, 0),                                  # 24 right goal-line, top touchline
            (L, (W - pb_w) / 2),                     # 25
            (L, (W - gb_w) / 2),                     # 26
            (L, (W + gb_w) / 2),                     # 27
            (L, (W + pb_w) / 2),                     # 28
            (L, W),                                  # 29 right goal-line, bottom touchline
            (L / 2 - r, W / 2),                      # 30 centre circle, left
            (L / 2 + r, W / 2),                      # 31 centre circle, right
        ]

    @property
    def vertices_m(self) -> np.ndarray:
        """The 32 landmarks in metres, shape (32, 2)."""
        return np.array(self.vertices, dtype=np.float32) / 100.0

    # Line segments between landmarks, for drawing the radar.
    edges: Sequence[Tuple[int, int]] = field(
        default_factory=lambda: (
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (6, 7),
            (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
            (17, 18), (18, 19), (19, 20), (22, 23),
            (24, 25), (25, 26), (26, 27), (27, 28), (28, 29),
            (0, 13), (1, 9), (2, 6), (3, 7), (4, 12), (5, 16),
            (13, 24), (17, 25), (22, 26), (23, 27), (20, 28), (16, 29),
        )
    )

    # ---- regions, in metres -------------------------------------------------

    def penalty_box_m(self, side: str) -> Tuple[float, float, float, float]:
        """(x_min, y_min, x_max, y_max) of a penalty area, in metres."""
        pb_l = self.penalty_box_length / 100.0
        y0 = (self.width - self.penalty_box_width) / 200.0
        y1 = (self.width + self.penalty_box_width) / 200.0
        if side == "left":
            return 0.0, y0, pb_l, y1
        return self.length_m - pb_l, y0, self.length_m, y1

    def goal_mouth_m(self, side: str) -> Tuple[np.ndarray, np.ndarray]:
        """The two posts of a goal, in metres."""
        half = self.goal_width / 200.0
        cy = self.width_m / 2.0
        x = 0.0 if side == "left" else self.length_m
        return np.array([x, cy - half]), np.array([x, cy + half])

    def in_penalty_box(self, point: Sequence[float], side: str) -> bool:
        x0, y0, x1, y1 = self.penalty_box_m(side)
        return x0 <= point[0] <= x1 and y0 <= point[1] <= y1

    def is_inside(self, point: Sequence[float], margin: float = 0.0) -> bool:
        return (
            -margin <= point[0] <= self.length_m + margin
            and -margin <= point[1] <= self.width_m + margin
        )


PITCH = PitchConfig()


# ---- coordinate conversions -------------------------------------------------

def metres_to_tagger(points: np.ndarray, pitch: PitchConfig = PITCH) -> np.ndarray:
    """Pitch metres -> the tagger's 120 x 80 grid.

    A plain axis-wise rescale. The tagger's grid is not metric (it is 120 x 80
    for a pitch that is nearer 105 x 68), so x and y scale by different factors;
    that is intended and matches how every already-stored match was recorded.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    out = np.empty_like(pts)
    out[:, 0] = pts[:, 0] / pitch.length_m * TAGGER_LENGTH
    out[:, 1] = pts[:, 1] / pitch.width_m * TAGGER_WIDTH
    return out


def tagger_to_metres(points: np.ndarray, pitch: PitchConfig = PITCH) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    out = np.empty_like(pts)
    out[:, 0] = pts[:, 0] / TAGGER_LENGTH * pitch.length_m
    out[:, 1] = pts[:, 1] / TAGGER_WIDTH * pitch.width_m
    return out


def mirror_metres(points: np.ndarray, pitch: PitchConfig = PITCH) -> np.ndarray:
    """Rotate a point 180 degrees about the centre spot.

    Used to normalise the second half so that a team always attacks the same
    way in the data, however the sides were swapped at the interval.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    out = np.empty_like(pts)
    out[:, 0] = pitch.length_m - pts[:, 0]
    out[:, 1] = pitch.width_m - pts[:, 1]
    return out
