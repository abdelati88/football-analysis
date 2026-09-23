"""Checks on cutting events out of a match as short clips.

The planning is what needs testing: which stretches of video get cut, and where
they start. Writing the files is OpenCV's job and needs a real video.
"""
import pytest

from football_ai.clips import Clip, ClipConfig, plan

FPS = 25.0
TOTAL = 25 * 600   # a ten-minute match


def event(mins, secs, kind="Shot", player="#9", confidence=0.7, **extra):
    return {
        "Mins": mins, "Secs": secs, "Event": kind, "Player": player,
        "Team": "Team 1", "confidence": confidence, **extra,
    }


def test_only_the_wanted_kinds_are_cut():
    """A match holds hundreds of passes and nobody opens hundreds of files."""
    events = [event(0, 10, "Pass"), event(0, 20, "Shot"), event(0, 30, "Dribble")]
    clips = plan(events, FPS, TOTAL)
    assert len(clips) == 1
    assert clips[0].events[0]["Event"] == "Shot"


def test_an_empty_kind_set_takes_everything():
    events = [event(0, 10, "Pass"), event(0, 40, "Dribble")]
    clips = plan(events, FPS, TOTAL, ClipConfig(kinds=frozenset()))
    assert len(clips) == 2


def test_the_clip_starts_before_the_event():
    """An event is the end of something. Starting at the moment it is recorded
    shows the consequence and hides the cause."""
    clips = plan([event(1, 0)], FPS, TOTAL, ClipConfig(lead_seconds=6, trail_seconds=3))
    clip = clips[0]
    moment = int(60 * FPS)
    assert clip.start_frame == moment - int(6 * FPS)
    assert clip.end_frame == moment + int(3 * FPS)
    assert moment - clip.start_frame > clip.end_frame - moment


def test_events_close_together_become_one_clip():
    """Three files covering the same four seconds is worse than one, and makes
    a passage of play look like three unrelated fragments."""
    events = [event(0, 30), event(0, 32, "Cross"), event(0, 33, "Tackle")]
    clips = plan(events, FPS, TOTAL)
    assert len(clips) == 1
    assert len(clips[0].events) == 3


def test_events_far_apart_stay_separate():
    clips = plan([event(0, 10), event(2, 0)], FPS, TOTAL)
    assert len(clips) == 2


def test_merging_can_be_switched_off():
    events = [event(0, 30), event(0, 32, "Cross")]
    clips = plan(events, FPS, TOTAL, ClipConfig(merge_overlapping=False))
    assert len(clips) == 2


def test_a_clip_never_starts_before_the_video_does():
    clips = plan([event(0, 1)], FPS, TOTAL, ClipConfig(lead_seconds=6))
    assert clips[0].start_frame == 0


def test_a_clip_never_runs_past_the_end():
    clips = plan([event(9, 59)], FPS, TOTAL, ClipConfig(trail_seconds=10))
    assert clips[0].end_frame <= TOTAL - 1


def test_the_exact_frame_is_preferred_over_the_clock():
    """Analysed events carry the frame they happened on; hand-tagged ones only
    carry minutes and seconds, so both have to work."""
    exact = plan([event(0, 0, frame=1234)], FPS, TOTAL, ClipConfig(lead_seconds=0, trail_seconds=0))
    assert exact[0].start_frame == 1234


def test_the_cap_keeps_the_best_supported_events():
    """A cap that took the first N would silently discard the second half of
    a match."""
    weak = [event(m, 0, confidence=0.3) for m in range(0, 5)]
    strong = [event(m, 0, confidence=0.9) for m in range(5, 10)]
    clips = plan(weak + strong, FPS, TOTAL, ClipConfig(max_clips=5))
    kept = [e for c in clips for e in c.events]
    assert len(kept) == 5
    assert all(e["confidence"] == 0.9 for e in kept)


def test_the_filename_says_what_is_in_the_clip():
    clips = plan([event(3, 7, "Shot", player="#9")], FPS, TOTAL)
    label = clips[0].label()
    assert "03m07s" in label and "Shot" in label and "9" in label
    assert all(c.isalnum() or c in "_-." for c in label)


def test_a_merged_clip_is_named_for_everything_in_it():
    events = [event(0, 30, "Cross"), event(0, 31, "Shot")]
    label = plan(events, FPS, TOTAL)[0].label()
    assert "Cross" in label and "Shot" in label


def test_nothing_to_cut_is_no_clips():
    assert plan([], FPS, TOTAL) == []
    assert plan([event(0, 10, "Pass")], FPS, TOTAL) == []


def test_merging_does_not_run_away():
    """Events every few seconds chain into each other. Unchecked, a busy
    passage merges back into the whole match — the first real run of this
    produced one "clip" covering the entire video."""
    events = [event(0, s) for s in range(10, 120, 4)]
    clips = plan(events, FPS, TOTAL, ClipConfig(max_clip_seconds=20))
    assert len(clips) > 1
    for clip in clips:
        assert clip.duration_s(FPS) <= 20 + 1e-6


def test_a_short_burst_still_merges():
    """The cap must not defeat the merging it is guarding."""
    events = [event(0, 30), event(0, 32, "Cross")]
    clips = plan(events, FPS, TOTAL, ClipConfig(max_clip_seconds=45))
    assert len(clips) == 1
