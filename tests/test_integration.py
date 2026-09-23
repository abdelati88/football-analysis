"""End-to-end check of the reasoning half of the pipeline.

Feeds fabricated but realistic tracking data — a passage of play with known
passes, a turnover and a shot — through possession, event derivation and
statistics, and checks the numbers that come out the other side. No model, no
video: this is the part that has to be right regardless of how good the
detector is.
"""
import numpy as np
import pandas as pd
import pytest

from football_ai.detect import GOALKEEPER, PLAYER, REFEREE
from football_ai.events import EventBuilder, EventConfig, infer_attacking_directions
from football_ai.pitch import PITCH
from football_ai.possession import (
    PossessionConfig, assign_ball_holder, build_spells, possession_share,
)
from football_ai.stats import summarise
from football_ai.track import track_classes

FPS = 25.0
STEP = 2                      # analysed every other frame, as the pipeline does


def build_scene():
    """A minute of play: Home pass down the right, a turnover, an Away shot.

    Home (team 1) attacks towards x = 105; Away (team 2) towards x = 0.
    """
    people_rows = []
    ball_rows = []

    # Static supporting cast, so the team/direction inference has something to
    # read: Home's keeper at the left end, Away's at the right.
    cast = [
        (101, 1, PLAYER, 30.0, 20.0), (102, 1, PLAYER, 40.0, 50.0),
        (103, 1, PLAYER, 55.0, 30.0), (104, 1, PLAYER, 60.0, 44.0),
        (105, 1, GOALKEEPER, 4.0, 34.0),
        (201, 2, PLAYER, 75.0, 22.0), (202, 2, PLAYER, 70.0, 48.0),
        (203, 2, PLAYER, 62.0, 36.0), (204, 2, PLAYER, 58.0, 30.0),
        (205, 2, GOALKEEPER, 101.0, 34.0),
        (300, 0, REFEREE, 52.0, 60.0),
    ]

    # The ball's carriers, in order, with where they stand and when they hold it.
    # (track_id, team, x, y, first_frame, last_frame)
    script = [
        (11, 1, 30.0, 34.0,   0,  30),      # Home 11 on the ball
        (12, 1, 52.0, 22.0,  44,  74),      # receives a pass
        (13, 1, 74.0, 20.0,  88, 118),      # and another, into the final third
        (21, 2, 75.4, 21.0, 126, 160),      # Away 21 takes it off his toe
        (22, 2, 26.0, 34.0, 176, 200),      # carries it into range
    ]

    for track_id, team, x, y, f0, f1 in script:
        for frame in range(f0, f1 + 1, STEP):
            people_rows.append((frame, frame / FPS, track_id, PLAYER,
                                0, 0, 0, 0, 0.0, 0.0, x, y))

    # Everyone else is on the pitch for the whole passage.
    for track_id, team, cls, x, y in cast:
        for frame in range(0, 260, STEP):
            people_rows.append((frame, frame / FPS, track_id, cls,
                                0, 0, 0, 0, 0.0, 0.0, x, y))

    # The ball: with each carrier while they hold it, travelling in between.
    def travel(f_from, f_to, a, b):
        span = max(f_to - f_from, 1)
        for frame in range(f_from, f_to, STEP):
            t = (frame - f_from) / span
            ball_rows.append((frame, frame / FPS,
                              a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))

    for i, (_, _, x, y, f0, f1) in enumerate(script):
        for frame in range(f0, f1 + 1, STEP):
            ball_rows.append((frame, frame / FPS, x, y))
        if i + 1 < len(script):
            nxt = script[i + 1]
            travel(f1 + STEP, nxt[4], (x, y), (nxt[2], nxt[3]))

    # Away 22 shoots at the goal it attacks (x = 0) and scores.
    for frame, x in zip(range(202, 226, STEP), np.linspace(26, -0.4, 12)):
        ball_rows.append((frame, frame / FPS, float(x), 34.0))

    people = pd.DataFrame(people_rows, columns=[
        "frame", "time_s", "track_id", "cls", "x1", "y1", "x2", "y2",
        "img_x", "img_y", "pitch_x", "pitch_y",
    ])
    ball = pd.DataFrame(ball_rows, columns=["frame", "time_s", "pitch_x", "pitch_y"])
    ball["img_x"] = ball["pitch_x"] * 10
    ball["img_y"] = ball["pitch_y"] * 10
    ball["conf"] = 0.9

    teams = {t: tm for t, tm, *_ in [(s[0], s[1]) for s in script]}
    teams.update({t: tm for t, tm, _, _, _ in cast if tm})
    return people, ball, teams


@pytest.fixture(scope="module")
def scene():
    people, ball, teams = build_scene()
    classes = track_classes(people)
    config = PossessionConfig()
    holder = assign_ball_holder(people, ball, teams, classes, config, frame_width=1920)
    spells = build_spells(holder, people, ball, teams, classes, config)
    directions = infer_attacking_directions(people, teams, classes, PITCH)
    events = EventBuilder(
        team_names={1: "Home", 2: "Away"}, directions=directions,
        config=EventConfig(), fps=FPS,
    ).build(spells, ball)
    return dict(people=people, ball=ball, teams=teams, classes=classes,
                holder=holder, spells=spells, events=events, directions=directions)


def test_attacking_directions_come_from_the_goalkeepers(scene):
    assert scene["directions"] == {1: 1, 2: -1}


def test_every_carrier_becomes_a_possession_spell(scene):
    carriers = [s.track_id for s in scene["spells"]]
    # The five scripted carriers, in order. The keeper collecting the ball out
    # of the net afterwards is a sixth, and correctly so.
    assert carriers[:5] == [11, 12, 13, 21, 22]


def test_the_passes_between_teammates_are_recognised(scene):
    passes = [e for e in scene["events"] if e.event == "Pass" and e.outcome == "Successful"]
    assert {p.player for p in passes} >= {"#11", "#12"}
    home_passes = [p for p in passes if p.team == 1]
    assert len(home_passes) == 2
    assert home_passes[0].start_m == pytest.approx((30.0, 34.0))
    assert home_passes[0].end_m == pytest.approx((52.0, 22.0))


def test_the_turnover_is_attributed_to_the_player_who_won_it(scene):
    won = [e for e in scene["events"] if e.event in ("Tackle", "Interception")]
    assert won, "the turnover produced no defensive event"
    # Won a metre and a half from the carrier: a tackle, not a read pass.
    assert won[0].event == "Tackle"
    assert won[0].team == 2 and won[0].player == "#21"


def test_the_shot_at_the_attacked_goal_is_a_goal(scene):
    shots = [e for e in scene["events"] if e.event == "Shot"]
    assert len(shots) == 1
    assert shots[0].player == "#22"
    assert shots[0].outcome == "Goal"


def test_statistics_agree_with_the_events(scene):
    possession = possession_share(scene["holder"], scene["teams"], PossessionConfig())
    stats = summarise(
        events=scene["events"], people=scene["people"], holder_frame=scene["holder"],
        teams=scene["teams"], classes=scene["classes"],
        team_names={1: "Home", 2: "Away"}, possession=possession, fps=FPS,
    )
    home, away = stats["teams"][1], stats["teams"][2]

    assert away["goals"] == 1 and home["goals"] == 0
    assert home["passes"] >= 2 and home["pass_accuracy"] > 0
    assert round(home["possession"] + away["possession"]) == 100
    # The referee is on the pitch throughout and must never appear as a player.
    assert all(p["role"] != "referee" for p in stats["players"])
    assert stats["totals"]["events"] == len(scene["events"])


def test_events_land_inside_the_taggers_grid(scene):
    for event in scene["events"]:
        row = event.as_tagger_row()
        assert 0 <= row["X"] <= 120 and 0 <= row["Y"] <= 80
        assert row["source"] == "ai"
        if row["X2"] != "":
            assert -1 <= row["X2"] <= 121
