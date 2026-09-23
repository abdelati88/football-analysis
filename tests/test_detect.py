"""Checks on the ball search geometry.

The ball is twelve pixels wide in a broadcast frame and every event depends on
where it is, so the windows the specialist model looks through have to cover
the frame completely and overlap at their seams. A ball that falls in a gap is
not merely missed — the possession it was about to decide gets attributed to
whoever happens to be standing near the last sighting.
"""
import numpy as np
import pytest

from football_ai.detect import tile_windows

FRAME = (1920, 1080)
SIZE = 640
OVERLAP = 128


def covers(width, height, size, overlap):
    """Mark every pixel some window sees, and return the unseen count."""
    seen = np.zeros((height, width), bool)
    for x0, y0, x1, y1 in tile_windows(width, height, size, overlap):
        seen[y0:y1, x0:x1] = True
    return (~seen).sum()


def test_the_windows_cover_the_whole_frame():
    assert covers(*FRAME, SIZE, OVERLAP) == 0


@pytest.mark.parametrize("width,height", [
    (1920, 1080), (1280, 720), (3840, 2160), (1000, 1000), (641, 641),
])
def test_every_frame_shape_is_covered(width, height):
    """Including sizes the step does not divide evenly, which is most of them."""
    assert covers(width, height, SIZE, OVERLAP) == 0


def test_windows_are_the_size_the_model_expects():
    """Not resized on the way in — that is the entire point of tiling rather
    than downscaling the frame — so every window must be exactly full size."""
    for x0, y0, x1, y1 in tile_windows(*FRAME, SIZE, OVERLAP):
        assert (x1 - x0, y1 - y0) == (SIZE, SIZE)


def test_neighbouring_windows_actually_overlap():
    """A ball on a boundary is a half-ball in both neighbours unless they
    share a strip wide enough to hold it whole."""
    tiles = tile_windows(*FRAME, SIZE, OVERLAP)
    xs = sorted({t[0] for t in tiles})
    ys = sorted({t[1] for t in tiles})
    for axis in (xs, ys):
        for a, b in zip(axis, axis[1:]):
            assert b - a <= SIZE - 32, "seam too wide to hold a ball whole"


def test_a_frame_smaller_than_one_window_gives_one_window():
    assert tile_windows(400, 300, SIZE, OVERLAP) == [(0, 0, 400, 300)]


def test_the_count_stays_affordable():
    """This path runs whenever the ball has been missing for a few frames, so
    the window count is a runtime figure, not just a geometry one."""
    assert len(tile_windows(*FRAME, SIZE, OVERLAP)) <= 9
