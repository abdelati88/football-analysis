"""Checks on kit classification — in particular the case that broke it once:
a team playing in the same colour family as the grass.
"""
import cv2
import numpy as np
import pytest

from football_ai.teams import TeamClassifier, TeamConfig, _jersey_pixels, estimate_grass_colour

GRASS = (58, 112, 45)          # BGR, a plausible pitch


def scene(kit_bgr, box=(180, 100, 220, 200)):
    """A frame of grass with one player-shaped block of kit colour on it."""
    frame = np.zeros((400, 400, 3), np.uint8)
    frame[:] = GRASS
    # Some variation, so the median is not a single exact value.
    noise = np.random.default_rng(0).integers(-6, 7, frame.shape, dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    x1, y1, x2, y2 = box
    # Torso occupies the upper part of the box, which is what gets sampled.
    frame[y1 + 12 : y1 + int((y2 - y1) * 0.5), x1 + 10 : x2 - 10] = kit_bgr
    return frame, [x1, y1, x2, y2]


def test_grass_colour_is_measured_from_the_frame():
    frame, _ = scene((255, 255, 255))
    measured = estimate_grass_colour(frame)
    assert measured is not None
    assert np.allclose(measured, GRASS, atol=12)


def test_a_green_kit_is_not_mistaken_for_grass():
    """The bug this guards against: removing every green pixel as 'pitch'
    deletes a lime kit entirely and leaves nothing to classify."""
    lime = (120, 250, 170)
    frame, box = scene(lime)
    grass = estimate_grass_colour(frame)
    kit = _jersey_pixels(frame, box, TeamConfig(), grass)
    assert kit is not None
    assert len(kit) > 100
    # What survives must be the kit, not the turf.
    assert np.allclose(np.median(kit, axis=0), lime, atol=25)


def test_separates_a_white_kit_from_a_lime_kit():
    white = (235, 235, 235)
    lime = (120, 250, 170)
    samples = []
    for _ in range(10):
        f, b = scene(white); samples.append((f, [b]))
        f, b = scene(lime);  samples.append((f, [b]))

    classifier = TeamClassifier().fit(samples)
    colours = np.array([classifier.team_colors[1], classifier.team_colors[2]], float)
    # The two learned colours must be far apart, and each near one real kit.
    assert np.linalg.norm(colours[0] - colours[1]) > 60
    for kit in (np.array(white, float), np.array(lime, float)):
        assert min(np.linalg.norm(colours - kit, axis=1)) < 45


def test_a_track_is_labelled_by_majority_not_by_one_frame():
    white, lime = (235, 235, 235), (120, 250, 170)
    samples = []
    for _ in range(10):
        f, b = scene(white); samples.append((f, [b]))
        f, b = scene(lime);  samples.append((f, [b]))
    classifier = TeamClassifier().fit(samples)

    # Track 1 is seen in white nine times and, once, misread as the other kit.
    for i in range(9):
        f, b = scene(white)
        classifier.observe(f, [(1, b)])
    f, b = scene(lime)
    classifier.observe(f, [(1, b)])

    for _ in range(6):
        f, b = scene(lime)
        classifier.observe(f, [(2, b)])

    assignments = classifier.finalise()
    assert assignments[1] != assignments[2]
    assert classifier.confidence_of(1) == pytest.approx(0.9)


def test_goalkeepers_join_the_team_defending_their_end():
    # Team 1 plays towards the right, so its outfield players average high x
    # and its keeper stands low.
    outfield = (
        [(i, 1, np.array([70.0, 34.0])) for i in range(10, 20)]
        + [(i, 2, np.array([35.0, 34.0])) for i in range(20, 30)]
    )
    keepers = [(1, np.array([3.0, 34.0])), (2, np.array([102.0, 34.0]))]
    assigned = TeamClassifier.assign_goalkeepers(keepers, outfield)
    assert assigned[1] == 1     # near x=0, furthest from team 1's outfield
    assert assigned[2] == 2


def test_fit_refuses_when_there_is_nothing_to_learn_from():
    with pytest.raises(ValueError, match="usable jersey samples"):
        TeamClassifier().fit([])
