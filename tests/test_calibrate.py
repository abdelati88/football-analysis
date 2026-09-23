"""Checks on the homography: does it recover real geometry, and does it refuse
the geometry it cannot trust?
"""
import cv2
import numpy as np
import pytest

from football_ai.calibrate import (
    CalibrationConfig, CalibrationTrack, Homography, _build, manual_homography,
)
from football_ai.pitch import PITCH

FRAME = (1920, 1080)


def synthetic_view(indices, jitter=0.0, seed=0):
    """Project chosen pitch landmarks through a plausible broadcast homography."""
    pitch_pts = PITCH.vertices_m[list(indices)]
    # A camera looking at the pitch from the side and above: the far touchline
    # is compressed, the near one is not.
    src = np.float32([[0, 0], [PITCH.length_m, 0], [PITCH.length_m, PITCH.width_m], [0, PITCH.width_m]])
    dst = np.float32([[300, 300], [1620, 300], [1850, 980], [70, 980]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    image_pts = cv2.perspectiveTransform(
        pitch_pts.astype(np.float32).reshape(-1, 1, 2), matrix
    ).reshape(-1, 2)
    if jitter:
        image_pts = image_pts + np.random.default_rng(seed).normal(0, jitter, image_pts.shape)
    return image_pts.astype(np.float32), pitch_pts.astype(np.float32)


def test_recovers_metres_from_clean_landmarks():
    indices = [0, 5, 24, 29, 13, 16, 8, 21]
    image_pts, pitch_pts = synthetic_view(indices)
    h = _build(image_pts, pitch_pts, 0, FRAME, PITCH, 8.0)
    assert h is not None
    back = h.to_pitch(image_pts)
    assert np.allclose(back, pitch_pts, atol=0.05)


def test_survives_realistic_detection_jitter():
    indices = [0, 5, 24, 29, 13, 16, 9, 12, 17, 20]
    image_pts, pitch_pts = synthetic_view(indices, jitter=2.5, seed=1)
    h = _build(image_pts, pitch_pts, 0, FRAME, PITCH, 8.0)
    assert h is not None
    error = np.linalg.norm(h.to_pitch(image_pts) - pitch_pts, axis=1)
    # A couple of pixels of landmark noise must stay under a metre on the pitch.
    assert error.mean() < 1.0


def test_rejects_collinear_landmarks():
    # Every point on one goal line: numerically solvable, physically nonsense.
    indices = [0, 1, 2, 3, 4, 5]
    image_pts, pitch_pts = synthetic_view(indices)
    assert _build(image_pts, pitch_pts, 0, FRAME, PITCH, 8.0) is None


def test_rejects_too_few_points():
    image_pts, pitch_pts = synthetic_view([0, 5, 24])
    assert _build(image_pts, pitch_pts, 0, FRAME, PITCH, 8.0) is None


def test_manual_homography_from_four_clicked_corners():
    indices = [0, 24, 29, 5]           # the four pitch corners
    image_pts, _ = synthetic_view(indices)
    h = manual_homography(image_pts, indices, FRAME)
    centre = h.to_pitch(h.to_image([[PITCH.length_m / 2, PITCH.width_m / 2]]))[0]
    assert centre == pytest.approx([PITCH.length_m / 2, PITCH.width_m / 2], abs=0.01)


def test_manual_homography_reports_unusable_points():
    with pytest.raises(ValueError):
        manual_homography([[0, 0], [1, 1], [2, 2], [3, 3]], [0, 1, 2, 3], FRAME)


def _fake(frame_index):
    identity = np.eye(3, dtype=np.float64)
    return Homography(frame_index, identity, identity, 8, 1.0)


def test_track_reuses_a_recent_solution_but_not_a_stale_one():
    track = CalibrationTrack(CalibrationConfig(max_reuse_frames=50))
    track.add(_fake(100))
    track.add(_fake(400))

    assert track.at(100) is not None
    assert track.at(120) is not None          # 20 frames on: the camera has barely moved
    assert track.at(250) is None              # 150 frames from either: too stale to trust
    assert track.at(390) is not None


def test_a_manual_homography_overrides_the_model():
    track = CalibrationTrack(CalibrationConfig())
    track.add(_fake(10))
    track.static = _fake(0)
    assert track.at(99999) is track.static


def _panning_view(shift_px: float):
    """The same camera, panned sideways by `shift_px`."""
    src = np.float32([[0, 0], [PITCH.length_m, 0], [PITCH.length_m, PITCH.width_m], [0, PITCH.width_m]])
    dst = np.float32([[300 + shift_px, 300], [1620 + shift_px, 300],
                      [1850 + shift_px, 980], [70 + shift_px, 980]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    inverse = np.linalg.inv(matrix)
    # matrix maps pitch -> image here, so swap the roles for the Homography.
    return Homography(0, inverse, matrix, 8, 1.0)


def test_interpolating_between_solutions_tracks_a_panning_camera():
    """Between two solved frames the blended mapping must land on the camera's
    real position, not on whichever neighbour happened to be nearer."""
    track = CalibrationTrack(CalibrationConfig(max_reuse_frames=50))
    start, end = _panning_view(0.0), _panning_view(200.0)
    start.frame_index, end.frame_index = 100, 120
    track.add(start)
    track.add(end)

    truth = _panning_view(100.0)          # where the camera actually is at 110
    blended = track.at(110)
    assert blended is not None

    centre = np.array([[PITCH.length_m / 2, PITCH.width_m / 2]], dtype=np.float32)
    expected = truth.to_image(centre)[0]
    got = blended.to_image(centre)[0]
    assert np.linalg.norm(got - expected) < 2.0

    # Snapping to the nearer solution would be out by half the pan.
    nearest = start.to_image(centre)[0]
    assert np.linalg.norm(nearest - expected) > 50.0


def test_interpolation_can_be_switched_off():
    track = CalibrationTrack(CalibrationConfig(max_reuse_frames=50, interpolate=False))
    start, end = _panning_view(0.0), _panning_view(200.0)
    start.frame_index, end.frame_index = 100, 120
    track.add(start)
    track.add(end)
    assert track.at(105) is start


def test_an_exact_hit_returns_that_solution_untouched():
    track = CalibrationTrack(CalibrationConfig())
    solved = _panning_view(0.0)
    solved.frame_index = 100
    track.add(solved)
    assert track.at(100) is solved
