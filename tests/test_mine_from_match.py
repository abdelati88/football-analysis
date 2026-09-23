"""Checks on finding the ball detector's mistakes without any labels.

The rule is that a ball cannot teleport: a detection far from the frame before
it, far from the frame after it, while those two are close to each other, is
not a ball. Everything here guards the ways that rule could go wrong — and the
most damaging way is not missing a phantom, it is mistaking a kick for one.
A kick is where the ball changes direction, so mining kicks would teach the
model to go blind at exactly the moment that matters.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "scripts"))

from mine_from_match import Sighting, find_teleports, median_step  # noqa: E402


def track(points, start=0, step=2):
    """A sighting per point, evenly spaced in time. `None` means no ball."""
    return [
        Sighting(frame=start + i * step,
                 xy=None if p is None else np.array(p, dtype=float),
                 conf=0.8,
                 box=None if p is None else np.array([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], float))
        for i, p in enumerate(points)
    ]


def test_a_ball_moving_steadily_has_no_teleports():
    assert find_teleports(track([(x, 100) for x in range(0, 200, 10)])) == []


def test_a_detection_that_jumps_and_comes_back_is_caught():
    points = [(0, 100), (10, 100), (20, 100), (900, 700), (40, 100), (50, 100)]
    found = find_teleports(track(points))
    assert len(found) == 1
    assert found[0].xy.tolist() == [900, 700]


def test_the_expected_position_is_between_the_neighbours():
    """The crop is labelled from this, so an error here writes the ball's
    label onto empty grass — or worse, omits it where the ball really is."""
    points = [(0, 100), (10, 100), (20, 100), (900, 700), (40, 100), (50, 100)]
    found = find_teleports(track(points))
    assert found[0].expected == pytest.approx([30.0, 100.0])


def test_a_kick_is_not_a_teleport():
    """The whole point. At a kick the ball is far from the frame before it —
    and the frames either side are far from each other too, because it really
    moved. Mining these would train the model to miss every strike."""
    points = [(0, 100), (10, 100), (20, 100), (120, 100), (220, 100), (320, 100)]
    assert find_teleports(track(points), factor=3.0) == []


def test_a_direction_reversal_is_not_a_teleport():
    """A ball played back the way it came passes close to where it was, but
    the neighbours are far apart, which is what separates it from a phantom."""
    points = [(0, 100), (50, 100), (100, 100), (150, 100), (100, 100), (50, 100)]
    assert find_teleports(track(points), factor=1.5) == []


def test_the_threshold_follows_the_footage():
    """A tight broadcast angle moves the ball across more pixels than a wide
    tactical one. A fixed pixel threshold would have to be retuned per video."""
    slow = track([(x, 100) for x in range(0, 100, 5)])
    fast = track([(x, 100) for x in range(0, 1000, 50)])
    assert median_step(fast) == pytest.approx(10 * median_step(slow))

    # The same absolute jump is impossible in slow footage, ordinary in fast.
    # Slow footage steps 5 px, so 6x is 30; fast steps 50 px, so 6x is 300.
    # A 100 px jump sits between them and must be read differently in each.
    jump = [(0, 100), (5, 100), (10, 100), (110, 100), (20, 100), (25, 100)]
    assert len(find_teleports(track(jump), scale=median_step(slow))) == 1
    assert find_teleports(track(jump), scale=median_step(fast)) == []


def test_frames_far_apart_in_time_are_left_alone():
    """After a long gap with no ball, the next sighting is allowed to be
    anywhere — the ball had a second to move, and calling that impossible
    would mine every reacquisition."""
    points = [(0, 100), (10, 100), (900, 700), (20, 100), (30, 100)]
    sightings = track(points)
    sightings[2].frame = sightings[1].frame + 40   # a long gap before it
    sightings[3].frame = sightings[2].frame + 40   # and after
    assert find_teleports(sightings) == []


def test_missing_sightings_do_not_crash_or_count():
    points = [(0, 100), None, (20, 100), (900, 700), None, (40, 100)]
    assert find_teleports(track(points)) == []


def test_nothing_to_look_at():
    assert find_teleports([]) == []
    assert find_teleports(track([(0, 0)])) == []
    assert median_step([]) == 0.0


def test_a_track_that_never_moves_yields_no_scale_and_no_findings():
    """A zero median step would make every threshold zero and flag the lot."""
    assert find_teleports(track([(5, 5)] * 8)) == []
