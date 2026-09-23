# Measuring against 97 hand-tagged events — 2026-09-14

## The correction that has to come first

The headline figure reported when this run finished — **63% recall, 26%
precision** — is substantially inflated, and the way to see it is to shuffle
the prediction times and score again:

    as measured                    recall 63%   precision 26%
    same events, times shuffled    recall 49%   precision 20%

A system that knows *nothing* — the same 237 events scattered at random across
the same six minutes — scores 49% recall. With 237 predictions, 97 truth events
and a ±4 s tolerance, predictions land next to a truth event of the same kind
and team by luck alone, and the more of them there are the more often it
happens.

So the shipped system's real signal was **+14 points of recall and +6 points of
precision above chance**, not 63% and 26%.

This also invalidates comparisons made on raw score: a configuration emitting
237 events collects more accidental credit than one emitting 130, so raw recall
rewards whichever guesses most. Every configuration below is scored against a
shuffle of *itself*.

## The diagnosis

Possession finds the right moments. It cuts them into far too many pieces.

    100% of hand-tagged events have a possession spell ending within 2 s
     90% within 1 s

Nothing is being missed geometrically. But six minutes of play produced:

    258 possession spells   (a real six minutes holds perhaps 130)
    121 turnovers           (a person tagging it saw about 20)

Every spurious turnover becomes an interception in the event list, which is why
the analysis reported 50 interceptions where 4 were tagged, and 35 dribbles
where 2 were.

## The fix

A turnover is a bigger claim than a pass, and it now needs more evidence: an
opponent must lead for `steal_frames` where a team-mate needs `confirm_frames`.
The asymmetry is a fact about football rather than a tuning knob — in a crowd
the player nearest the ball alternates between the two teams frame by frame,
and only one of those alternations is a real turnover.

Alongside it, `regain_frames`: a player who loses the ball to *nobody* and has
it back within a blink never lost it, and without that the gap became two
possessions with a fabricated event between them.

    configuration          events  recall  chance   lift   prec  chance   lift
    shipped                   237     63%     49%   +14%    26%     20%    +6%
    turnover needs proof      140     51%     33%   +18%    35%     23%   +12%

Raw recall falls from 63% to 51% and that fall is almost entirely lost chance
credit: the lift *rises*, at every tolerance tested (4 s, 2 s and 1 s).
Precision lift doubles.

    turnovers   121 -> 51
    spells      258 -> 135
    dribbles     35 -> 27
    interceptions 50 -> 22

## What was tried and rejected

**Filtering by the confidence the system already computes.** It carries real
signal — matched events have a median confidence of 0.60 against 0.46 for
invented ones — and a cut at 0.55 lifts F1 from 0.365 to 0.506. It was rejected
because it discards real events to hide fabricated ones: recall falls from 63%
to 42%. It treats the symptom. The cause is that the fabricated events are
generated at all.

**Removing the `current == -1` shortcut outright.** On 9 events (2026-09-03)
this collapsed recall from 67% to 22% and the conclusion recorded then was that
the shortcut is load-bearing. On 97 events it is one of the better settings.
**That earlier conclusion was an artefact of a nine-event truth set** — exactly
the failure the refusal to tune on nine events was meant to avoid, and it still
reached the written conclusion.

## Still open

Shots and crosses are barely found: 0 of 3 shots and 0 of 5 crosses matched,
and these are the events a club actually watches. The analysis emits shots and
crosses, so it is placing them wrongly rather than failing to consider them.
That is the next thing to measure, and it is a different layer — event
classification, not possession.
