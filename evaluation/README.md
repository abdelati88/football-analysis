# Event accuracy — the only measurement a customer asks for

Everything else in this repository measures *detection*: the ball is found in
94% of frames, players score 0.987 mAP50, the pitch solves in every frame. None
of those answers whether the passes are passes. A detector can be perfect while
the football is wrong.

This directory holds the first honest attempt at that number.

## The run of 2026-09-02 — a 30-second clip, `08fd33_4.mp4`

    truth_08fd33_4.json       9 events, tagged by hand
    predicted_08fd33_4.json  19 events, from the analysis
    report_08fd33_4.json     the scored comparison

    recall           78%   (7 of the 9 tagged events were found)
    precision        37%   over all kinds
                     78%   over the kinds that were actually tagged
    outcome agreed   71%   of the matched events
    timing            1.0 s median

**Read the two precision figures as one fact, not two.** The hand-tagging
covered Pass and Clearance only. The analysis also reported Dribbles,
Interceptions and a Cross — ten events judged "invented" for the sole reason
that no ground truth for those kinds exists. Restricting both sides to the
kinds that were tagged is the like-for-like comparison, and gives 78%.

Neither figure is the product's precision. The first is a floor that assumes
every untagged kind is wrong; the second is a ceiling that declines to look at
them. The honest statement is that Pass and Clearance were measured, and
Dribble, Interception and Cross were not measured at all.

## The sample is nine events

At n=9, one event moves recall by eleven points. Nothing here supports a claim
to a customer. It establishes the instrument works end to end and gives a first
reading; ten minutes of tagged play is what would make the number mean
something.

## What the disagreements actually were

    0:16  tagged Clearance/T2     analysis said Interception/T2
    0:17  tagged Clearance/T1     analysis said Pass/T2

Two adjacent events where the analysis disagreed about *who had the ball*, not
about whether something happened. That is possession attribution — the ball is
handed to the nearest player rather than the one who touched it — and it is a
different defect from a missed event, with a different fix.

## Reproducing

    .venv/bin/python training/scripts/evaluate_events.py \
        --truth evaluation/truth_08fd33_4.json \
        --predicted evaluation/predicted_08fd33_4.json

Team names must match on both sides; the matcher requires agreement on kind and
team, so renaming a team scores every event as a miss for a reason that has
nothing to do with accuracy.
