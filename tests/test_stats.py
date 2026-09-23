"""Checks on the physical statistics, where tracking noise does the most damage."""
import numpy as np
import pandas as pd
import pytest

from football_ai.stats import MAX_PLAYER_SPEED_MS, heatmap, physical_stats

FPS = 25.0


def track(track_id, xs, ys):
    n = len(xs)
    return pd.DataFrame({
        "frame": range(n),
        "time_s": np.arange(n) / FPS,
        "track_id": [track_id] * n,
        "cls": [2] * n,
        "pitch_x": xs,
        "pitch_y": ys,
    })


def test_distance_matches_a_straight_run():
    # 2 m/s for 100 frames = 4 s = 8 m.
    xs = [i * (2.0 / FPS) for i in range(101)]
    stats = physical_stats(track(1, xs, [34.0] * 101), FPS)[1]
    assert stats["distance_m"] == pytest.approx(8.0, abs=0.05)
    assert stats["top_speed_kmh"] == pytest.approx(7.2, abs=0.2)


def test_an_identity_swap_does_not_add_phantom_distance():
    # Ten metres of real running, then the tracker jumps the id 40 m across the
    # pitch in a single frame, then ten more metres of running.
    real = [i * (2.0 / FPS) for i in range(126)]          # 10 m
    jumped = [x + 40.0 for x in real]                      # same run, shifted
    xs = real + jumped
    stats = physical_stats(track(1, xs, [34.0] * len(xs)), FPS)[1]
    assert stats["distance_m"] == pytest.approx(20.0, abs=0.2)
    assert stats["top_speed_kmh"] <= MAX_PLAYER_SPEED_MS * 3.6


def test_single_frame_jitter_does_not_set_the_top_speed():
    xs = [50.0] * 50
    xs[25] = 51.5  # one frame, 1.5 m off, then back
    stats = physical_stats(track(1, xs, [34.0] * 50), FPS)[1]
    # Averaged over the speed window this is well under a sprint.
    assert stats["top_speed_kmh"] < 20.0


def test_average_position_and_minutes():
    xs = [10.0] * 750 + [30.0] * 750   # 60 s at 25 fps
    stats = physical_stats(track(1, xs, [34.0] * 1500), FPS)[1]
    assert stats["avg_position"][0] == pytest.approx(20.0)
    assert stats["minutes"] == pytest.approx(1.0, abs=0.01)


def test_heatmap_is_normalised_and_correctly_oriented():
    # Everything in the bottom-left corner of the pitch.
    pts = np.column_stack([np.full(50, 3.0), np.full(50, 3.0)])
    grid = heatmap(pts, bins=(12, 8))
    assert len(grid) == 12 and len(grid[0]) == 8
    assert grid[0][0] == 1.0
    assert sum(sum(row) for row in grid) == 1.0


def test_empty_input_is_handled():
    empty = pd.DataFrame(columns=["frame", "time_s", "track_id", "cls", "pitch_x", "pitch_y"])
    assert physical_stats(empty, FPS) == {}
    assert heatmap(np.empty((0, 2)))[0][0] == 0.0
