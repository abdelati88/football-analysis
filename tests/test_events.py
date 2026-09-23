"""Synthetic checks on the event derivation.

These build possession spells by hand, so they test the football reasoning on
its own — no model, no video, no tracking noise.
"""
import numpy as np
import pandas as pd
import pytest

from football_ai.events import (
    CLEARANCE, CROSS, DRIBBLE, GOAL, INTERCEPTION, PASS, SHOT, SHOT_ASSIST,
    SUCCESSFUL, TACKLE, UNSUCCESSFUL, EventBuilder, EventConfig,
)
from football_ai.possession import Spell
from football_ai.pitch import PITCH

FPS = 25.0
TEAM_NAMES = {1: "Home", 2: "Away"}
# Home attacks towards x = 105, Away towards x = 0.
DIRECTIONS = {1: 1, 2: -1}


def spell(track_id, team, f0, f1, start, end, ball_start=None, ball_end=None, gk=False):
    return Spell(
        track_id=track_id, team=team, start_frame=f0, end_frame=f1,
        start_time=f0 / FPS, end_time=f1 / FPS,
        start_xy=np.array(start, float), end_xy=np.array(end, float),
        ball_start_xy=np.array(ball_start or start, float),
        ball_end_xy=np.array(ball_end or end, float),
        is_goalkeeper=gk,
    )


def ball_df(points, first_frame=0):
    """points: list of (frame, x, y) in pitch metres."""
    rows = [(f, f / FPS, x, y) for f, x, y in points]
    return pd.DataFrame(rows, columns=["frame", "time_s", "pitch_x", "pitch_y"])


def build(spells, ball, **cfg):
    return EventBuilder(TEAM_NAMES, DIRECTIONS, config=EventConfig(**cfg)).build(spells, ball)


def test_completed_pass_between_teammates():
    spells = [
        spell(7, 1, 100, 110, (40, 30), (40, 30)),
        spell(9, 1, 120, 130, (58, 22), (58, 22)),
    ]
    ball = ball_df([(110, 40, 30), (115, 49, 26), (120, 58, 22)])
    events = build(spells, ball)
    passes = [e for e in events if e.event == PASS]
    assert len(passes) == 1
    p = passes[0]
    assert p.outcome == SUCCESSFUL
    assert p.team == 1 and p.player == "#7"
    assert p.start_m == pytest.approx((40, 30))
    assert p.end_m == pytest.approx((58, 22))


def test_pass_to_an_opponent_is_unsuccessful_and_an_interception():
    spells = [
        spell(7, 1, 100, 110, (40, 30), (40, 30)),
        spell(4, 2, 125, 140, (62, 24), (62, 24)),
    ]
    ball = ball_df([(110, 40, 30), (118, 51, 27), (125, 62, 24)])
    events = build(spells, ball)
    kinds = [(e.event, e.outcome, e.player) for e in events]
    assert (PASS, UNSUCCESSFUL, "#7") in kinds
    assert (INTERCEPTION, "", "#4") in kinds


def test_close_quick_turnover_is_a_tackle_not_an_interception():
    spells = [
        spell(7, 1, 100, 110, (40, 30), (40, 30)),
        spell(4, 2, 114, 130, (41.5, 31), (41.5, 31)),
    ]
    ball = ball_df([(110, 40, 30), (114, 41.5, 31)])
    events = build(spells, ball)
    assert [e.event for e in events] == [TACKLE]
    assert events[0].player == "#4" and events[0].team == 2


def test_shot_that_crosses_the_line_between_the_posts_is_a_goal():
    spells = [spell(11, 1, 200, 205, (90, 34), (90, 34))]
    # Ball runs from the penalty spot straight into the middle of the goal.
    ball = ball_df([(205, 90, 34), (208, 97, 34), (211, 105.2, 34)])
    spells.append(spell(1, 2, 240, 250, (2, 34), (2, 34), gk=True))
    events = build(spells, ball)
    shots = [e for e in events if e.event == SHOT]
    assert len(shots) == 1
    assert shots[0].outcome == GOAL
    assert shots[0].player == "#11"


def test_shot_saved_by_the_keeper():
    spells = [
        spell(11, 1, 200, 205, (88, 32), (88, 32)),
        spell(1, 2, 212, 230, (104, 34), (104, 34), gk=True),
    ]
    ball = ball_df([(205, 88, 32), (208, 96, 33), (212, 104, 34)])
    events = build(spells, ball)
    shots = [e for e in events if e.event == SHOT]
    assert len(shots) == 1 and shots[0].outcome == "Saved"


def test_a_slow_sideways_ball_near_goal_is_a_pass_not_a_shot():
    spells = [
        spell(11, 1, 200, 205, (88, 20), (88, 20)),
        spell(12, 1, 240, 250, (88, 48), (88, 48)),
    ]
    ball = ball_df([(205, 88, 20), (220, 88, 34), (240, 88, 48)])
    events = build(spells, ball)
    assert [e.event for e in events] == [PASS]


def test_wide_delivery_into_the_box_is_a_cross():
    spells = [
        spell(2, 1, 300, 305, (92, 4), (92, 4)),
        spell(9, 1, 320, 330, (96, 34), (96, 34)),
    ]
    ball = ball_df([(305, 92, 4), (312, 94, 20), (320, 96, 34)])
    events = build(spells, ball)
    assert [e.event for e in events] == [CROSS]
    assert events[0].outcome == SUCCESSFUL


def test_long_ball_out_of_defence_to_an_opponent_is_a_clearance():
    spells = [
        spell(5, 1, 400, 405, (12, 34), (12, 34)),
        spell(8, 2, 440, 450, (60, 40), (60, 40)),
    ]
    ball = ball_df([(405, 12, 34), (420, 36, 37), (440, 60, 40)])
    events = build(spells, ball)
    assert [e.event for e in events] == [CLEARANCE]


def test_running_with_the_ball_is_a_dribble():
    spells = [
        spell(10, 1, 500, 540, (40, 34), (52, 34)),
        spell(10, 1, 545, 560, (52, 34), (52, 34)),
    ]
    ball = ball_df([(f, 40 + (f - 500) * 0.3, 34) for f in range(500, 561, 5)])
    events = build(spells, ball)
    dribbles = [e for e in events if e.event == DRIBBLE]
    assert len(dribbles) == 1
    assert dribbles[0].outcome == SUCCESSFUL
    assert "12.0 m" in dribbles[0].note


def test_the_pass_that_sets_up_a_shot_becomes_a_shot_assist():
    spells = [
        spell(8, 1, 100, 105, (70, 40), (70, 40)),
        spell(9, 1, 118, 124, (88, 34), (88, 34)),
    ]
    ball = ball_df(
        [(105, 70, 40), (112, 79, 37), (118, 88, 34), (124, 88, 34), (127, 96, 34), (130, 105.3, 34)]
    )
    events = build(spells, ball)
    kinds = [e.event for e in events]
    assert SHOT_ASSIST in kinds and SHOT in kinds
    assist = next(e for e in events if e.event == SHOT_ASSIST)
    assert assist.player == "#8"


def test_spells_separated_by_a_long_gap_produce_no_event():
    spells = [
        spell(7, 1, 100, 110, (40, 30), (40, 30)),
        spell(9, 1, 600, 610, (58, 22), (58, 22)),
    ]
    ball = ball_df([(110, 40, 30), (600, 58, 22)])
    assert build(spells, ball) == []


def test_tagger_row_uses_the_120_by_80_grid():
    spells = [
        spell(7, 1, 100, 110, (52.5, 34), (52.5, 34)),
        spell(9, 1, 120, 130, (70, 34), (70, 34)),
    ]
    ball = ball_df([(110, 52.5, 34), (120, 70, 34)])
    row = build(spells, ball)[0].as_tagger_row()
    assert row["X"] == 60.0 and row["Y"] == 40.0   # centre spot
    assert row["Mins"] == 0 and row["Secs"] == 4
    assert row["source"] == "ai"


def test_second_half_events_are_rotated_like_hand_tagged_ones():
    """The manual tagger stores second-half events rotated about the centre
    spot. Detected events must use the same convention or the two sources would
    be drawn in opposite halves of the same pitch."""
    spells = [
        spell(7, 1, 100, 110, (26.25, 17.0), (26.25, 17.0)),
        spell(9, 1, 120, 130, (40.0, 20.0), (40.0, 20.0)),
    ]
    ball = ball_df([(110, 26.25, 17.0), (120, 40.0, 20.0)])

    first = EventBuilder(TEAM_NAMES, DIRECTIONS, config=EventConfig(), half=1).build(spells, ball)
    second = EventBuilder(TEAM_NAMES, DIRECTIONS, config=EventConfig(), half=2).build(spells, ball)

    a = first[0].as_tagger_row()
    b = second[0].as_tagger_row()

    # A quarter of the way along and a quarter across, in the first half.
    assert (a["X"], a["Y"]) == pytest.approx((30.0, 20.0))
    # The same physical spot, recorded from the other end.
    assert (b["X"], b["Y"]) == pytest.approx((90.0, 60.0))
    assert b["half"] == 2
    assert (b["X2"], b["Y2"]) == pytest.approx((120 - a["X2"], 80 - a["Y2"]))


def test_positions_are_clamped_to_the_taggers_pitch():
    """Calibration error, or a ball genuinely over the line, can place an event
    outside the grid. The tagger's canvas is the pitch: nothing can be drawn
    beyond it, so nothing should be recorded beyond it."""
    spells = [
        spell(7, 1, 100, 110, (110.0, 72.0), (110.0, 72.0)),   # past both lines
        spell(9, 1, 120, 130, (-4.0, -3.0), (-4.0, -3.0)),
    ]
    ball = ball_df([(110, 110.0, 72.0), (120, -4.0, -3.0)])
    # Opposite corners is the only way to exercise both clamps at once, and it
    # is also further than the ball can be kicked — which a separate rule
    # rejects. Lift that limit here so this test measures the one thing it is
    # about; the plausibility rule has its own tests.
    row = build(spells, ball, max_pass_distance_m=400.0)[0].as_tagger_row()
    assert 0 <= row["X"] <= 120 and 0 <= row["Y"] <= 80
    assert 0 <= row["X2"] <= 120 and 0 <= row["Y2"] <= 80
    assert row["X"] == 120.0 and row["Y"] == 80.0
    assert row["X2"] == 0.0 and row["Y2"] == 0.0


def test_a_ball_further_than_anyone_can_kick_it_is_not_a_pass():
    """Two possessions eighty metres apart are not joined by a kick — the
    tracker lost the ball between them. Recording the pass that would have had
    to connect them invents a piece of football that did not happen.

    Found in the wild: a 74.7 m "pass" in the last second of a real clip, which
    the evidence score had already marked the weakest event of the run but
    which was recorded anyway, because nothing checked the distance.
    """
    spells = [
        spell(7, 1, 100, 110, (12, 34), (12, 34)),
        spell(9, 1, 120, 130, (95, 34), (95, 34)),
    ]
    ball = ball_df([(110, 12, 34), (115, 54, 34), (120, 95, 34)])
    events = build(spells, ball)
    assert not [e for e in events if e.event in (PASS, CROSS, CLEARANCE)]


def test_a_keeper_is_allowed_a_longer_ball_than_an_outfielder():
    """The one player who does clear seventy metres."""
    receiver = spell(9, 1, 120, 130, (80, 34), (80, 34))
    ball = ball_df([(110, 10, 34), (115, 45, 34), (120, 80, 34)])

    outfield = spell(7, 1, 100, 110, (10, 34), (10, 34))
    keeper = spell(7, 1, 100, 110, (10, 34), (10, 34), gk=True)

    assert not [e for e in build([outfield, receiver], ball) if e.event == PASS]
    assert [e for e in build([keeper, receiver], ball) if e.event == PASS]
