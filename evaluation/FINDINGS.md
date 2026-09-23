# What the measurement loop found — 2026-09-03

## The correction that matters first

Yesterday's figure of **78% recall / 78% precision was measured against a stored
analysis run from an older model state.** Re-running the current pipeline on the
same clip gives 18 events instead of 19, with different track numbers.

The current, correct baseline is:

    recall     67%
    precision  67%   (on the kinds that were hand-tagged)

The pipeline is **fully deterministic** — two cache runs produced byte-identical
player and ball frames — so this is not run-to-run noise. It is a real
difference between two versions of the models, and it means any figure quoted
must name the model state it was measured on.

## The hypothesis was wrong

The stated plan was to fix possession attribution: "the ball is handed to the
nearest player rather than the one who touched it." Tracing the disputed
seconds frame by frame showed something more specific — and the obvious fix for
it makes the system much worse.

`_smooth_holder` requires a new candidate to lead for `confirm_frames` before
possession switches, *except* when nobody currently holds the ball:

    if candidate_run >= config.confirm_frames or current == -1:

Since the ball is unheld most of the time (in flight), that exception is the
normal path. **Twelve of fourteen possession handovers in the clip took the
shortcut; only two were confirmed.** The guard whose comment describes stopping
"a ball flying past a defender" does not run in the case it was written for.

Removing the shortcut is the obvious repair, and it is wrong:

    confirm=4, shortcut on   (shipped)   recall 67%   precision 67%
    confirm=4, shortcut off              recall 22%   precision 50%
    confirm=6, shortcut off              recall 22%   precision 67%

Recall collapses. At 12.5 fps a real touch does not hold the ball within 2.4 m
for four consecutive sampled frames — football touches are shorter than 0.32 s.
The shortcut is load-bearing: the phantom spells are the price of catching the
real ones, and the price is worth paying at this frame rate.

A second candidate — reject spells where the ball's trajectory never changed,
since a ball passing in a straight line was not touched — does not separate
either. Almost every spell contains some three-frame window with a large
apparent turn, because the projected position of an airborne ball is noisy.

## The thresholds are not the lever

54 combinations of `max_distance_m`, `confirm_frames`, `release_frames` and
`min_spell_frames` were scored. **None beat the shipped defaults.** The defaults
sit on a flat plateau, and the remaining errors come from somewhere other than
these four numbers — most likely team assignment and the event-classification
rules, which is where to look next.

## Why this stops here

Nine hand-tagged events. One event moves recall by eleven points. A change that
improved this score could not be distinguished from a change that fitted it, so
no change is being made to the possession layer on this evidence.

The blocker is not ideas. It is that the truth set is nine events long, covers
two event kinds of five, and comes from thirty seconds of play.

## What was built

    training/scripts/cache_tracks.py     freeze detection/tracking/calibration
    training/scripts/possession_lab.py   re-derive + score in ~1 second

The lab reproduces the shipping pipeline **exactly** — event for event, on the
same clip — which was checked rather than assumed. Detection takes 60 s and the
layers above it take milliseconds, so this turns a one-minute experiment into a
one-second one. That is what makes it possible to test fifty-four settings
instead of arguing about four.
