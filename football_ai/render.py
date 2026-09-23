"""Drawing the analysis back onto the video.

Two views, composited into one frame: the broadcast image with players marked
in their team's colour, and a top-down radar showing where everyone actually is
on the pitch. The radar is the part a coach reads — it is the whole point of
having recovered the homography — and it doubles as an honest display of the
calibration, because if the mapping is wrong the players visibly leave the
pitch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import cv2
import numpy as np

from .detect import BALL, GOALKEEPER, PLAYER, REFEREE
from .pitch import PITCH, PitchConfig

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
REFEREE_COLOUR = (0, 220, 255)   # BGR amber
BALL_COLOUR = (255, 255, 255)
PITCH_GREEN = (58, 112, 45)
LINE_GREY = (225, 232, 235)


@dataclass
class RenderConfig:
    radar_width: int = 420
    radar_margin: int = 18
    radar_opacity: float = 0.82
    show_track_ids: bool = True
    show_radar: bool = True
    show_hud: bool = True
    line_thickness: int = 2


def _contrast_text(bgr: Sequence[int]) -> tuple[int, int, int]:
    """Black or white, whichever stays readable on `bgr`."""
    b, g, r = (float(c) for c in bgr)
    luma = 0.114 * b + 0.587 * g + 0.299 * r
    return BLACK if luma > 140 else WHITE


def draw_player(
    frame: np.ndarray,
    bbox: Sequence[float],
    colour: Sequence[int],
    label: str | None = None,
    has_ball: bool = False,
    thickness: int = 2,
) -> None:
    """An ellipse on the ground under the player, not a box around them.

    A box tracks the bounding rectangle of a moving body and jitters with every
    raised arm; an ellipse at the feet sits still, reads as a position on the
    pitch, and does not hide the player.
    """
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    width = max(x2 - x1, 4)
    centre = ((x1 + x2) // 2, y2)
    colour = tuple(int(c) for c in colour)

    cv2.ellipse(
        frame, centre, (int(width * 0.55), int(width * 0.2)),
        0.0, -35, 250, colour, thickness, lineType=cv2.LINE_AA,
    )

    if has_ball:
        # A marker above the head of whoever is judged to be in possession.
        tip = (centre[0], y1 - 6)
        pts = np.array([tip, (tip[0] - 11, tip[1] - 16), (tip[0] + 11, tip[1] - 16)], np.int32)
        cv2.drawContours(frame, [pts], 0, colour, cv2.FILLED, lineType=cv2.LINE_AA)
        cv2.drawContours(frame, [pts], 0, BLACK, 1, lineType=cv2.LINE_AA)

    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        bx1, by1 = centre[0] - tw // 2 - 5, y2 + 5
        bx2, by2 = centre[0] + tw // 2 + 5, y2 + th + 13
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), colour, cv2.FILLED)
        cv2.putText(
            frame, label, (bx1 + 5, by2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, _contrast_text(colour), 1, cv2.LINE_AA,
        )


def draw_ball(frame: np.ndarray, centre: Sequence[float], predicted: bool = False) -> None:
    """Mark the ball: a solid ring where it was seen, a dashed one where it was not.

    The ball is missed on about one frame in ten — behind a leg, blurred to
    nothing by a hard pass — and those frames are filled in from the sightings
    either side. Saying so matters: a viewer who cannot tell an observation
    from an interpolation cannot tell how much to trust what they are watching.

    The distinction used to be the small filled centre, present when observed
    and absent when not. At this size that reads as the marker glitching rather
    than as a change of state — the first person to watch the output described
    it as "a small part disappeared and came back". A dashed ring cannot be
    mistaken for a flicker: it is plainly a different way of drawing the same
    thing, which is what "estimated" should look like.
    """
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    if predicted:
        _dashed_circle(frame, (cx, cy), 9, BLACK, 3)
        _dashed_circle(frame, (cx, cy), 9, BALL_COLOUR, 2)
        return
    cv2.circle(frame, (cx, cy), 9, BLACK, 3, lineType=cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 9, BALL_COLOUR, 2, lineType=cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 3, BALL_COLOUR, cv2.FILLED, lineType=cv2.LINE_AA)


def _dashed_circle(
    frame: np.ndarray,
    centre: tuple[int, int],
    radius: int,
    colour: Sequence[int],
    thickness: int,
    dashes: int = 6,
) -> None:
    """A circle drawn as `dashes` arcs with gaps between them."""
    span = 360.0 / dashes
    for i in range(dashes):
        start = i * span
        cv2.ellipse(
            frame, centre, (radius, radius), 0.0,
            start, start + span * 0.55,
            tuple(int(c) for c in colour), thickness, lineType=cv2.LINE_AA,
        )


def render_pitch(
    width: int,
    pitch: PitchConfig = PITCH,
    padding: int = 22,
) -> tuple[np.ndarray, float, int]:
    """A blank top-down pitch. Returns the image, its metres-to-pixels scale and padding."""
    scale = (width - 2 * padding) / pitch.length_m
    height = int(round(pitch.width_m * scale)) + 2 * padding
    img = np.full((height, width, 3), PITCH_GREEN, np.uint8)

    # Mown stripes, purely so the radar reads as a pitch at a glance.
    stripe = (width - 2 * padding) // 10
    for i in range(10):
        if i % 2:
            x0 = padding + i * stripe
            img[padding:height - padding, x0:x0 + stripe] = (
                np.array(PITCH_GREEN, np.uint8) + np.array([6, 10, 6], np.uint8)
            )

    def to_px(pt: Sequence[float]) -> tuple[int, int]:
        return int(round(padding + pt[0] * scale)), int(round(padding + pt[1] * scale))

    verts = pitch.vertices_m
    for a, b in pitch.edges:
        cv2.line(img, to_px(verts[a]), to_px(verts[b]), LINE_GREY, 2, cv2.LINE_AA)
    cv2.circle(
        img, to_px((pitch.length_m / 2, pitch.width_m / 2)),
        int(round(pitch.centre_circle_radius / 100.0 * scale)), LINE_GREY, 2, cv2.LINE_AA,
    )
    for spot in ((pitch.penalty_spot_distance / 100.0, pitch.width_m / 2),
                 (pitch.length_m - pitch.penalty_spot_distance / 100.0, pitch.width_m / 2),
                 (pitch.length_m / 2, pitch.width_m / 2)):
        cv2.circle(img, to_px(spot), 3, LINE_GREY, cv2.FILLED, cv2.LINE_AA)

    return img, scale, padding


def draw_radar(
    players: Iterable[tuple[np.ndarray, Sequence[int], str | None]],
    ball: np.ndarray | None,
    width: int,
    pitch: PitchConfig = PITCH,
) -> np.ndarray:
    """Top-down view. `players` is `(pitch_xy_metres, bgr_colour, label)`."""
    img, scale, padding = render_pitch(width, pitch)

    def to_px(pt: Sequence[float]) -> tuple[int, int]:
        return int(round(padding + pt[0] * scale)), int(round(padding + pt[1] * scale))

    for xy, colour, label in players:
        if xy is None or not np.all(np.isfinite(xy)):
            continue
        px = to_px(xy)
        cv2.circle(img, px, 7, BLACK, cv2.FILLED, cv2.LINE_AA)
        cv2.circle(img, px, 6, tuple(int(c) for c in colour), cv2.FILLED, cv2.LINE_AA)
        if label:
            cv2.putText(
                img, label, (px[0] - 6, px[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, WHITE, 1, cv2.LINE_AA,
            )

    if ball is not None and np.all(np.isfinite(ball)):
        px = to_px(ball)
        cv2.circle(img, px, 6, BLACK, cv2.FILLED, cv2.LINE_AA)
        cv2.circle(img, px, 4, BALL_COLOUR, cv2.FILLED, cv2.LINE_AA)

    return img


def overlay(frame: np.ndarray, panel: np.ndarray, margin: int, opacity: float) -> None:
    """Composite `panel` into the bottom-centre of `frame`, in place."""
    fh, fw = frame.shape[:2]
    ph, pw = panel.shape[:2]
    if pw >= fw or ph >= fh:
        return
    x0 = (fw - pw) // 2
    y0 = fh - ph - margin
    region = frame[y0:y0 + ph, x0:x0 + pw]
    cv2.addWeighted(panel, opacity, region, 1.0 - opacity, 0, region)
    cv2.rectangle(frame, (x0, y0), (x0 + pw, y0 + ph), (30, 30, 30), 2)


def draw_hud(
    frame: np.ndarray,
    team_names: Dict[int, str],
    team_colours: Dict[int, Sequence[int]],
    possession: Dict[int, float],
    clock: str,
) -> None:
    """A scoreboard-style strip: who is who, who has had the ball, and when."""
    fw = frame.shape[1]
    w, h = min(520, fw - 40), 78
    x0, y0 = 20, 20
    panel = frame[y0:y0 + h, x0:x0 + w]
    cv2.addWeighted(np.full_like(panel, (18, 18, 20)), 0.72, panel, 0.28, 0, panel)
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (60, 60, 65), 1)

    cv2.putText(frame, clock, (x0 + 14, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)

    # A single bar split by possession share, labelled with each team's name.
    bar_x, bar_y, bar_w, bar_h = x0 + 14, y0 + 40, w - 28, 20
    share1 = float(possession.get(1, 50.0))
    split = int(bar_w * max(0.0, min(100.0, share1)) / 100.0)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + split, bar_y + bar_h),
                  tuple(int(c) for c in team_colours.get(1, (200, 200, 200))), cv2.FILLED)
    cv2.rectangle(frame, (bar_x + split, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  tuple(int(c) for c in team_colours.get(2, (120, 120, 120))), cv2.FILLED)

    left = f"{team_names.get(1, 'Team 1')} {share1:.0f}%"
    right = f"{100 - share1:.0f}% {team_names.get(2, 'Team 2')}"
    cv2.putText(frame, left, (bar_x + 6, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                _contrast_text(team_colours.get(1, (200, 200, 200))), 1, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(frame, right, (bar_x + bar_w - tw - 6, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                _contrast_text(team_colours.get(2, (120, 120, 120))), 1, cv2.LINE_AA)


def format_clock(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
