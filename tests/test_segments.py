"""Checks on splitting a video into shots.

Two failures matter here and they are not symmetric. A missed cut lets the
tracker pair strangers and lets a homography from one camera place players
seen by another — silently wrong numbers. A false cut merely throws away
continuity that was real, which costs identities but invents nothing. So the
tests below lean on not firing during ordinary camera movement.
"""
import numpy as np
import pytest

from football_ai.segments import (
    SegmentConfig, Shot, ShotDetector, distance, signature,
)

SIZE = (270, 480, 3)


def scene(seed, brightness=110):
    """A frame of structured noise — stands in for one camera's view."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 90, (9, 16, 3), dtype=np.uint8) + brightness
    frame = np.repeat(np.repeat(base, 30, axis=0), 30, axis=1)
    return np.clip(frame, 0, 255).astype(np.uint8)


def panned(frame, pixels):
    """The same scene, shifted — what a camera move looks like."""
    return np.roll(frame, pixels, axis=1)


def test_a_single_shot_is_one_shot():
    detector = ShotDetector()
    base = scene(1)
    cuts = [detector.observe(i, panned(base, i * 3)) for i in range(40)]
    assert not any(cuts[1:])
    assert len(detector.shots) == 1


def test_an_edit_starts_a_new_shot():
    detector = ShotDetector()
    a, b = scene(1), scene(99)
    for i in range(20):
        detector.observe(i, panned(a, i * 2))
    cut = detector.observe(20, b)
    assert cut
    assert len(detector.shots) == 2
    assert detector.shots[1].start_frame == 20


def test_panning_is_not_an_edit():
    """The failure that would make this useless: firing on ordinary movement,
    which on a broadcast never stops."""
    detector = ShotDetector()
    base = scene(4)
    fired = [detector.observe(i, panned(base, i * 6)) for i in range(1, 60)]
    assert not any(fired)


def test_every_frame_belongs_to_a_shot():
    detector = ShotDetector()
    for i in range(15):
        detector.observe(i, scene(1 if i < 8 else 2))
    covered = sum(s.frames for s in detector.shots)
    assert covered == 15


def test_a_shot_where_the_pitch_is_never_found_is_not_play():
    """A replay or a close-up: continuous footage, but not the match."""
    config = SegmentConfig()
    close_up = Shot(index=0, start_frame=0, end_frame=40, frames=40,
                    attempted=20, calibrated=1)
    wide = Shot(index=1, start_frame=41, end_frame=200, frames=160,
                attempted=40, calibrated=38)
    assert not close_up.is_play(config)
    assert wide.is_play(config)


def test_a_flash_between_shots_is_not_play():
    """Wipes and graphics produce a handful of frames that are not football."""
    config = SegmentConfig()
    flash = Shot(index=0, start_frame=0, end_frame=2, frames=3,
                 attempted=3, calibrated=3)
    assert not flash.is_play(config)


def test_a_shot_nothing_was_tried_on_is_left_alone():
    """No calibration attempted is no evidence, not evidence of absence — the
    pitch model may simply be missing."""
    config = SegmentConfig()
    untested = Shot(index=0, start_frame=0, end_frame=100, frames=100)
    assert untested.is_play(config)


def test_distance_is_zero_for_identical_frames():
    frame = scene(7)
    config = SegmentConfig()
    assert distance(signature(frame, config), signature(frame, config)) == 0.0


def test_distance_grows_with_how_different_the_frames_are():
    config = SegmentConfig()
    base = scene(3)
    near = signature(panned(base, 4), config)
    far = signature(scene(500), config)
    origin = signature(base, config)
    assert distance(origin, near) < distance(origin, far)


def test_detection_can_be_switched_off():
    detector = ShotDetector(SegmentConfig(enabled=False))
    for i in range(10):
        detector.observe(i, scene(i))
    assert len(detector.shots) == 1


def test_the_summary_counts_cuts_not_shots():
    detector = ShotDetector()
    for i in range(10):
        detector.observe(i, scene(1))
    for i in range(10, 20):
        detector.observe(i, scene(2))
    summary = detector.summary()
    assert summary["shots"] == 2
    assert summary["cuts"] == 1
