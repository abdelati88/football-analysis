"""Checks on re-linking split identities.

The dangerous failure here is not a missed merge — that just leaves the
fragmentation we already had. It is a *wrong* merge, which invents a player who
teleports and hands one man's statistics to another. Most of what follows tests
the refusals.
"""
import numpy as np
import pandas as pd
import pytest

from football_ai.detect import GOALKEEPER, PLAYER, REFEREE
from football_ai.stitch import (
    StitchConfig, apply_mapping, build_tracklets, link, remap_dict, stitch,
)

FPS = 12.5
WIDTH = 1920.0


def run(start_frame, count, start_xy, velocity_m_s, track_id, cls=PLAYER, step=1):
    """One fragment of a player moving at a constant velocity, in metres."""
    rows = []
    per_frame = np.asarray(velocity_m_s, float) / FPS
    for i in range(count):
        frame = start_frame + i * step
        x, y = np.asarray(start_xy, float) + per_frame * (i * step)
        rows.append(
            {
                "frame": frame, "time_s": frame / FPS, "track_id": track_id, "cls": cls,
                "x1": 0.0, "y1": 0.0, "x2": 20.0, "y2": 50.0,
                # Image coordinates deliberately unrelated to the pitch ones:
                # nothing that follows may quietly depend on them.
                "img_x": 100.0 + i, "img_y": 200.0,
                "pitch_x": x, "pitch_y": y,
            }
        )
    return rows


def frame_of(*runs):
    return pd.DataFrame([row for r in runs for row in r])


def stitched(people, teams, classes=None, config=None):
    classes = classes or {int(t): PLAYER for t in people["track_id"].unique()}
    return stitch(
        people, teams, classes, kits=None, fps=FPS, frame_width=WIDTH,
        config=config or StitchConfig(),
    )


# ---------------------------------------------------------------- the merge


def test_a_player_split_across_a_gap_is_rejoined():
    """Track 1 runs to (30, 40) and dies; track 2 appears half a second later,
    two metres further along the same line. That is one player."""
    people = frame_of(
        run(0, 20, (20.0, 40.0), (6.0, 0.0), track_id=1),
        run(26, 20, (31.5, 40.0), (6.0, 0.0), track_id=2),
    )
    out, mapping, report = stitched(people, {1: 1, 2: 1})

    assert report.merges == 1
    assert mapping[1] == mapping[2]
    assert out["track_id"].nunique() == 1


def test_the_rejoined_track_keeps_every_observation():
    people = frame_of(
        run(0, 20, (20.0, 40.0), (6.0, 0.0), track_id=1),
        run(26, 20, (31.5, 40.0), (6.0, 0.0), track_id=2),
    )
    out, _, _ = stitched(people, {1: 1, 2: 1})
    assert len(out) == len(people)
    assert out["frame"].is_monotonic_increasing


def test_a_chain_of_three_fragments_collapses_to_one():
    people = frame_of(
        run(0, 15, (10.0, 30.0), (5.0, 0.0), track_id=1),
        run(20, 15, (17.0, 30.0), (5.0, 0.0), track_id=2),
        run(40, 15, (25.0, 30.0), (5.0, 0.0), track_id=3),
    )
    _, mapping, report = stitched(people, {1: 1, 2: 1, 3: 1})
    assert len({mapping[1], mapping[2], mapping[3]}) == 1
    assert report.tracks_after == 1


# ------------------------------------------------------------ the refusals


def test_fragments_seen_in_the_same_frame_are_never_merged():
    """The one rule that needs no threshold: a person is in one place at a
    time. Both fragments sit at the same spot, which without this rule is the
    most attractive merge there is."""
    people = frame_of(
        run(0, 30, (40.0, 30.0), (0.0, 0.0), track_id=1),
        run(10, 30, (40.0, 30.0), (0.0, 0.0), track_id=2),
    )
    _, mapping, report = stitched(people, {1: 1, 2: 1})
    assert report.merges == 0
    assert mapping[1] != mapping[2]


def test_opposing_players_are_never_merged():
    people = frame_of(
        run(0, 20, (20.0, 40.0), (6.0, 0.0), track_id=1),
        run(26, 20, (31.5, 40.0), (6.0, 0.0), track_id=2),
    )
    _, _, report = stitched(people, {1: 1, 2: 2})
    assert report.merges == 0


def test_a_referee_is_not_merged_into_a_player():
    people = frame_of(
        run(0, 20, (20.0, 40.0), (6.0, 0.0), track_id=1),
        run(26, 20, (31.5, 40.0), (6.0, 0.0), track_id=2, cls=REFEREE),
    )
    _, _, report = stitched(
        people, {1: 1, 2: 1}, classes={1: PLAYER, 2: REFEREE}
    )
    assert report.merges == 0


def test_a_keeper_is_not_merged_into_an_outfielder():
    people = frame_of(
        run(0, 20, (20.0, 40.0), (6.0, 0.0), track_id=1, cls=GOALKEEPER),
        run(26, 20, (31.5, 40.0), (6.0, 0.0), track_id=2),
    )
    _, _, report = stitched(
        people, {1: 1, 2: 1}, classes={1: GOALKEEPER, 2: PLAYER}
    )
    assert report.merges == 0


def test_a_gap_no_human_could_close_is_refused():
    """Half a second later and sixty metres away — the length of the pitch at
    120 m/s. Whoever that is, it is not the same man."""
    people = frame_of(
        run(0, 20, (20.0, 40.0), (0.0, 0.0), track_id=1),
        run(26, 20, (80.0, 40.0), (0.0, 0.0), track_id=2),
    )
    _, _, report = stitched(people, {1: 1, 2: 1})
    assert report.merges == 0


def test_the_speed_budget_scales_with_the_gap():
    """Eight metres is impossible in a fifth of a second and unremarkable in
    two, so the same displacement must be refused in one case and allowed in
    the other. A fixed distance threshold cannot express that."""
    near = frame_of(
        run(0, 20, (20.0, 40.0), (0.0, 0.0), track_id=1),
        run(22, 20, (28.0, 40.0), (0.0, 0.0), track_id=2),
    )
    far = frame_of(
        run(0, 20, (20.0, 40.0), (0.0, 0.0), track_id=1),
        run(45, 20, (28.0, 40.0), (0.0, 0.0), track_id=2),
    )
    assert stitched(near, {1: 1, 2: 1})[2].merges == 0
    assert stitched(far, {1: 1, 2: 1})[2].merges == 1


def test_a_gap_longer_than_the_last_round_is_left_alone():
    """Beyond the configured horizon the evidence is gone; the honest answer is
    two identities, not a guess."""
    people = frame_of(
        run(0, 20, (20.0, 40.0), (0.0, 0.0), track_id=1),
        run(200, 20, (21.0, 40.0), (0.0, 0.0), track_id=2),
    )
    _, _, report = stitched(people, {1: 1, 2: 1})
    assert report.merges == 0


# --------------------------------------------------------- global choice


def test_the_better_of_two_candidates_wins_globally():
    """Two fragments end at once and two begin at once, crossed over: the
    greedy first-come choice is wrong, and only judging all four together gets
    it right."""
    people = frame_of(
        run(0, 20, (20.0, 20.0), (0.0, 0.0), track_id=1),
        run(0, 20, (20.0, 55.0), (0.0, 0.0), track_id=2),
        run(26, 20, (22.0, 56.0), (0.0, 0.0), track_id=3),   # belongs to 2
        run(26, 20, (22.0, 21.0), (0.0, 0.0), track_id=4),   # belongs to 1
    )
    _, mapping, report = stitched(people, {1: 1, 2: 1, 3: 1, 4: 1})
    assert report.merges == 2
    assert mapping[1] == mapping[4]
    assert mapping[2] == mapping[3]
    assert mapping[1] != mapping[2]


def test_one_fragment_cannot_absorb_two_successors():
    people = frame_of(
        run(0, 20, (40.0, 40.0), (0.0, 0.0), track_id=1),
        run(26, 20, (41.0, 40.0), (0.0, 0.0), track_id=2),
        run(26, 20, (39.0, 40.0), (0.0, 0.0), track_id=3),
    )
    _, mapping, _ = stitched(people, {1: 1, 2: 1, 3: 1})
    assert mapping[2] != mapping[3]


# ------------------------------------------------------------ kit evidence


def test_a_grossly_different_kit_blocks_a_plausible_merge():
    people = frame_of(
        run(0, 20, (20.0, 40.0), (6.0, 0.0), track_id=1),
        run(26, 20, (31.5, 40.0), (6.0, 0.0), track_id=2),
    )
    kits = {1: np.zeros(15, np.float32), 2: np.full(15, 1.0, np.float32)}
    _, _, report = stitch(
        people, {1: 1, 2: 1}, {1: PLAYER, 2: PLAYER}, kits,
        fps=FPS, frame_width=WIDTH,
    )
    assert report.merges == 0


def test_matching_kits_leave_a_plausible_merge_alone():
    people = frame_of(
        run(0, 20, (20.0, 40.0), (6.0, 0.0), track_id=1),
        run(26, 20, (31.5, 40.0), (6.0, 0.0), track_id=2),
    )
    kits = {1: np.full(15, 0.4, np.float32), 2: np.full(15, 0.41, np.float32)}
    _, _, report = stitch(
        people, {1: 1, 2: 1}, {1: PLAYER, 2: PLAYER}, kits,
        fps=FPS, frame_width=WIDTH,
    )
    assert report.merges == 1


# ------------------------------------------------- uncalibrated footage


def test_uncalibrated_footage_falls_back_to_image_space():
    """No homography means no metres. Linking still has to do something, and
    what it does is judged in pixels — conservatively."""
    people = frame_of(
        run(0, 20, (20.0, 40.0), (0.0, 0.0), track_id=1),
        run(26, 20, (20.0, 40.0), (0.0, 0.0), track_id=2),
    )
    people[["pitch_x", "pitch_y"]] = np.nan
    _, _, report = stitched(people, {1: 1, 2: 1})
    assert report.merges == 1
    assert report.used_image_fallback


def test_image_fallback_can_be_switched_off():
    people = frame_of(
        run(0, 20, (20.0, 40.0), (0.0, 0.0), track_id=1),
        run(26, 20, (20.0, 40.0), (0.0, 0.0), track_id=2),
    )
    people[["pitch_x", "pitch_y"]] = np.nan
    config = StitchConfig(allow_image_fallback=False)
    _, _, report = stitched(people, {1: 1, 2: 1}, config=config)
    assert report.merges == 0


# ------------------------------------------------------------- plumbing


def test_disabled_stitching_changes_nothing():
    people = frame_of(
        run(0, 20, (20.0, 40.0), (6.0, 0.0), track_id=1),
        run(26, 20, (31.5, 40.0), (6.0, 0.0), track_id=2),
    )
    out, _, report = stitched(people, {1: 1, 2: 1}, config=StitchConfig(enabled=False))
    assert report.merges == 0
    assert out["track_id"].nunique() == 2


def test_an_empty_frame_is_handled():
    empty = pd.DataFrame(
        columns=["frame", "time_s", "track_id", "cls", "x1", "y1", "x2", "y2",
                 "img_x", "img_y", "pitch_x", "pitch_y"]
    )
    out, mapping, report = stitched(empty, {})
    assert out.empty and not mapping and report.merges == 0


def test_remap_dict_resolves_by_majority():
    """Two fragments say team 1 and one says team 2; the merged player is on
    team 1."""
    values = {1: 1, 2: 1, 3: 2}
    assert remap_dict(values, {1: 1, 2: 1, 3: 1}) == {1: 1}


def test_apply_mapping_leaves_unmapped_tracks_alone():
    people = frame_of(run(0, 10, (10.0, 10.0), (0.0, 0.0), track_id=7))
    out = apply_mapping(people, {1: 1})
    assert set(out["track_id"]) == {7}


def test_a_fragment_too_short_to_place_is_not_linked():
    config = StitchConfig(min_tracklet_obs=5)
    people = frame_of(
        run(0, 20, (20.0, 40.0), (0.0, 0.0), track_id=1),
        run(26, 2, (21.0, 40.0), (0.0, 0.0), track_id=2),
    )
    tracklets = build_tracklets(people, {1: 1, 2: 1}, {1: PLAYER, 2: PLAYER}, None, config)
    assert [t.track_id for t in tracklets] == [1]


def test_the_budget_tightens_as_the_gap_grows():
    """The two-regime budget in one assertion: a burst is allowed at once, but
    the allowance per second falls away afterwards, because net displacement
    over several seconds is nothing like a sprint held in a straight line."""
    from football_ai.stitch import _reach

    cfg = StitchConfig()
    per_second = [_reach(t, cfg) / t for t in (0.25, 1.0, 3.0, 6.0)]
    assert per_second == sorted(per_second, reverse=True)
    assert _reach(0.25, cfg) == pytest.approx(2.25)
