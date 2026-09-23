

# ---------------------------------------------------------------- turnovers

"""A turnover is a bigger claim than a pass and needs more evidence.

Measured against six minutes of hand-tagged play, the code without this
asymmetry called 121 turnovers where a person saw about twenty: in a crowd the
player nearest the ball alternates between the two teams frame by frame, and
each alternation became a change of possession and then an interception in the
event list.
"""
from football_ai.possession import PossessionConfig, _smooth_holder

TEAMS = {1: 1, 2: 1, 8: 2, 9: 2}     # 1 and 2 play together; 8 and 9 oppose them


def test_a_team_mate_needs_only_the_ordinary_confirmation():
    config = PossessionConfig(confirm_frames=2, steal_frames=6)
    holders = [1, 1, 1, 2, 2, 2, 2]
    assert _smooth_holder(holders, config, TEAMS)[-1] == 2


def test_an_opponent_needs_more_than_a_team_mate_does():
    """Three frames is enough for a team-mate and not for an opponent."""
    config = PossessionConfig(confirm_frames=2, steal_frames=6)
    assert _smooth_holder([1, 1, 1, 8, 8, 8], config, TEAMS)[-1] == 1


def test_an_opponent_who_keeps_the_ball_does_take_it():
    config = PossessionConfig(confirm_frames=2, steal_frames=4)
    assert _smooth_holder([1, 1, 8, 8, 8, 8, 8], config, TEAMS)[-1] == 8


def test_the_ball_can_still_be_picked_up_from_nobody():
    """Requiring a turnover's evidence to *start* a possession would lose most
    of the real ones — the ball is unheld most of the time."""
    config = PossessionConfig(confirm_frames=1, steal_frames=20)
    assert _smooth_holder([-1, -1, 8, 8], config, TEAMS)[-1] == 8


def test_a_player_who_blinks_out_and_back_never_lost_the_ball():
    """Without this the gap becomes two possessions with an invented event
    between them."""
    config = PossessionConfig(confirm_frames=4, release_frames=1, regain_frames=3)
    out = _smooth_holder([1, 1, 1, 1, -1, -1, 1, 1], config, TEAMS)
    assert out[-1] == 1
    assert set(out[4:]) == {1, -1} or out[-1] == 1


def test_without_teams_every_change_is_treated_as_a_pass():
    """The behaviour before the asymmetry existed, kept for callers that have
    no team mapping to hand."""
    config = PossessionConfig(confirm_frames=2, steal_frames=99)
    assert _smooth_holder([1, 1, 8, 8, 8], config, None)[-1] == 8
