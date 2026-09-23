"""Checks on the event-accuracy measurement itself.

This script produces the one number the project will be sold on, so it has to
be harder to fool than the thing it measures. The failure that matters is a
flattering one: a matcher that pairs anything with anything reports a system
that works perfectly and is worth nothing.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "scripts"))

from evaluate_events import hand_tagged_only, match_events, score   # noqa: E402


def event(mins, secs, kind="Pass", team="Team 1", outcome="Successful", **extra):
    return {"Mins": mins, "Secs": secs, "Event": kind, "Team": team,
            "Outcome": outcome, **extra}


TOL = 4.0


def test_identical_tagging_scores_perfectly():
    events = [event(0, 5), event(0, 20, "Shot"), event(1, 2, "Interception")]
    r = score(events, list(events), TOL)
    assert r["recall"] == 1.0 and r["precision"] == 1.0
    assert r["outcome_accuracy"] == 1.0


def test_a_late_tag_still_matches():
    """A person tagging live is a beat behind the play; that is not an error."""
    truth = [event(0, 10)]
    predicted = [event(0, 12)]
    assert score(truth, predicted, TOL)["recall"] == 1.0


def test_an_event_far_away_in_time_does_not_match():
    truth = [event(0, 10)]
    predicted = [event(0, 40)]
    r = score(truth, predicted, TOL)
    assert r["recall"] == 0.0 and r["precision"] == 0.0


def test_a_different_kind_never_matches():
    """The whole point is telling a pass from a dribble. Pairing them would
    report the confusion as a success."""
    truth = [event(0, 10, "Pass")]
    predicted = [event(0, 10, "Dribble")]
    assert score(truth, predicted, TOL)["matched"] == 0


def test_the_other_team_never_matches():
    truth = [event(0, 10, team="Team 1")]
    predicted = [event(0, 10, team="Team 2")]
    assert score(truth, predicted, TOL)["matched"] == 0


def test_one_prediction_cannot_satisfy_two_events():
    """Otherwise a single lucky guess covers a burst and recall is inflated."""
    truth = [event(0, 10), event(0, 12)]
    predicted = [event(0, 11)]
    r = score(truth, predicted, TOL)
    assert r["matched"] == 1
    assert r["recall"] == pytest.approx(0.5)


def test_a_burst_is_paired_by_best_total_fit_not_first_come():
    """Two passes a second apart, predicted a second apart. Greedy matching
    can pair them crossed over and inflate the timing error; the assignment
    should find the ordering that fits."""
    truth = [event(0, 10), event(0, 13)]
    predicted = [event(0, 11), event(0, 14)]
    pairs, missed, spurious = match_events(truth, predicted, TOL)
    assert sorted(pairs) == [(0, 0), (1, 1)]
    assert not missed and not spurious


def test_extra_predictions_cost_precision_not_recall():
    truth = [event(0, 10)]
    predicted = [event(0, 10), event(0, 30), event(1, 0)]
    r = score(truth, predicted, TOL)
    assert r["recall"] == 1.0
    assert r["precision"] == pytest.approx(1 / 3)


def test_outcome_is_scored_only_on_matched_events():
    """Getting the kind right and the outcome wrong is a real, smaller error,
    and should be visible as exactly that."""
    truth = [event(0, 10, outcome="Successful")]
    predicted = [event(0, 10, outcome="Unsuccessful")]
    r = score(truth, predicted, TOL)
    assert r["recall"] == 1.0
    assert r["outcome_accuracy"] == 0.0


def test_per_kind_figures_add_up():
    truth = [event(0, 5, "Pass"), event(0, 9, "Pass"), event(0, 20, "Shot")]
    predicted = [event(0, 5, "Pass"), event(0, 20, "Shot")]
    r = score(truth, predicted, TOL)
    assert r["per_kind"]["Pass"] == {"truth": 2, "predicted": 1, "matched": 1}
    assert r["per_kind"]["Shot"] == {"truth": 1, "predicted": 1, "matched": 1}


def test_ai_events_are_not_accepted_as_ground_truth():
    """A match holds both hand-tagged and imported events. Scoring an analysis
    against its own output would report a flawless system."""
    mixed = [event(0, 5, source="manual"), event(0, 9, source="ai"), event(0, 12)]
    kept = hand_tagged_only(mixed)
    assert len(kept) == 2
    assert all(e.get("source") != "ai" for e in kept)


def test_nothing_predicted_is_zero_recall_not_a_crash():
    r = score([event(0, 5)], [], TOL)
    assert r["recall"] == 0.0 and r["matched"] == 0
