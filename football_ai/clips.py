"""Cutting each event out of the match as its own short clip.

The annotated video is the proof that the analysis works, and nobody watches
it. That is fine for a thirty-second test and useless for the thing being sold:
a coach handed ninety minutes of footage with rings drawn on it has been handed
ninety minutes of footage. What a coach actually wants is the eleven shots, or
every time this one player lost the ball, and wants them in a couple of
minutes rather than an afternoon.

The analysis already knows when everything happened, to the frame. Turning that
into a clip per event is nearly free — one more pass over the video, writing
whichever frames fall inside a window — and it converts a technical output into
something that gets used on a Monday morning.

Two details decide whether the clips are watchable.

**The lead matters more than the trail.** An event is the *end* of something: a
pass is recorded where the ball leaves the foot, a tackle where possession
changes. Starting the clip at that moment shows the consequence and hides the
cause, which is the half a coach is looking for. So the window reaches further
back than forward.

**Overlapping events become one clip.** A dribble, the pass that ended it and
the interception that followed can occupy four seconds between them. Three
clips of the same four seconds is worse than one: it triples the file count,
and it makes a passage of play look like three unrelated fragments. Windows
that overlap are merged, and the resulting clip is labelled with everything
inside it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from . import video as vid


@dataclass
class ClipConfig:
    """How to cut events out of a match."""

    enabled: bool = False

    # An event is recorded at the moment it completes, so the build-up sits
    # before it. Weighted accordingly: a shot is worth seeing from the pass
    # that made it, not from the strike.
    lead_seconds: float = 6.0
    trail_seconds: float = 3.0

    # Which events are worth a clip. Everything is rarely the answer — a match
    # holds hundreds of passes and nobody opens hundreds of files — so the
    # default is the handful anyone would actually sit down to watch. An empty
    # set means every event.
    kinds: frozenset[str] = frozenset({"Shot", "Goal", "Cross", "Shot Assist", "Tackle"})

    # Clips whose windows overlap are merged into one.
    merge_overlapping: bool = True
    # But merging is not allowed to run away. Events every few seconds chain
    # into each other, and unchecked that turns a busy passage back into the
    # match — the first test of this produced a single "clip" covering the
    # whole video. Past this length a new clip starts even though the windows
    # touch, because something a coach has to scrub through is not a clip.
    max_clip_seconds: float = 45.0
    # A safety valve: a match with a generous `kinds` could otherwise fill a
    # disk. Events beyond this are dropped, strongest evidence kept.
    max_clips: int = 60


@dataclass
class Clip:
    """One stretch of video and the events inside it."""

    index: int
    start_frame: int
    end_frame: int
    events: List[dict] = field(default_factory=list)
    path: Path | None = None

    def duration_s(self, fps: float) -> float:
        return (self.end_frame - self.start_frame + 1) / fps if fps else 0.0

    def label(self) -> str:
        """A filename stem describing what is in the clip.

        Built from the events themselves, so the file is identifiable in a
        folder listing without opening it — which is the whole point of
        cutting them out in the first place.
        """
        if not self.events:
            return f"{self.index:03d}_clip"
        first = self.events[0]
        minute, second = int(first.get("Mins", 0)), int(first.get("Secs", 0))
        kinds = []
        for e in self.events:
            kind = str(e.get("Event", "")).strip()
            if kind and kind not in kinds:
                kinds.append(kind)
        who = str(first.get("Player") or "").strip().lstrip("#")
        parts = [f"{self.index:03d}", f"{minute:02d}m{second:02d}s", "-".join(kinds[:3])]
        if who:
            parts.append(who)
        stem = "_".join(p for p in parts if p)
        # Filenames end up in URLs and on other people's filesystems.
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", stem)[:80]

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "events": len(self.events),
            "kinds": sorted({str(e.get("Event", "")) for e in self.events}),
            "file": self.path.name if self.path else None,
        }


def _event_frame(event: dict, fps: float) -> int:
    """When the event happened, in source frames.

    `frame` is exact and present on anything the analysis produced; the
    minute/second pair is the fallback for hand-tagged events, which is why
    this accepts either.
    """
    frame = event.get("frame")
    if frame is not None:
        return int(frame)
    seconds = float(event.get("Mins", 0) or 0) * 60 + float(event.get("Secs", 0) or 0)
    return int(round(seconds * fps))


def plan(
    events: Sequence[dict],
    fps: float,
    total_frames: int,
    config: ClipConfig | None = None,
) -> List[Clip]:
    """Decide which stretches of video to cut, without touching the video."""
    config = config or ClipConfig()
    wanted = [
        e for e in events
        if not config.kinds or str(e.get("Event", "")) in config.kinds
    ]
    if not wanted:
        return []

    if len(wanted) > config.max_clips:
        # Keep the best-supported ones rather than the first N: a cap that
        # takes whatever happens to come early would silently discard the
        # second half of a match.
        wanted = sorted(
            wanted, key=lambda e: float(e.get("confidence", 0) or 0), reverse=True
        )[: config.max_clips]

    lead = int(round(config.lead_seconds * fps))
    trail = int(round(config.trail_seconds * fps))
    limit = max(total_frames - 1, 0) if total_frames else None

    windows = []
    for event in sorted(wanted, key=lambda e: _event_frame(e, fps)):
        centre = _event_frame(event, fps)
        start = max(centre - lead, 0)
        end = centre + trail
        if limit is not None:
            end = min(end, limit)
        windows.append((start, end, event))

    cap = int(round(config.max_clip_seconds * fps))
    clips: List[Clip] = []
    for start, end, event in windows:
        if (
            config.merge_overlapping
            and clips
            and start <= clips[-1].end_frame
            and max(clips[-1].end_frame, end) - clips[-1].start_frame <= cap
        ):
            clips[-1].end_frame = max(clips[-1].end_frame, end)
            clips[-1].events.append(event)
            continue
        clips.append(
            Clip(index=len(clips), start_frame=start, end_frame=end, events=[event])
        )
    for i, clip in enumerate(clips):
        clip.index = i
    return clips


def cut(
    video_path: str | Path,
    clips: Sequence[Clip],
    out_dir: str | Path,
    fps: float | None = None,
    on_progress=None,
) -> List[Clip]:
    """Write each planned clip out as its own file.

    One pass over the source. Seeking to each clip in turn would be simpler to
    read and much slower on a long match, because a seek to an arbitrary frame
    in a compressed video decodes from the previous keyframe anyway — doing
    that sixty times re-decodes most of the match sixty times.
    """
    if not clips:
        return []

    source = Path(video_path)
    info = vid.probe(source)
    rate = fps or info.fps
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)

    ordered = sorted(clips, key=lambda c: c.start_frame)
    first_frame = min(c.start_frame for c in ordered)
    last_frame = max(c.end_frame for c in ordered)

    writers: Dict[int, vid.VideoWriter] = {}
    done = 0
    for index, frame in vid.frames(source, start=first_frame, stop=last_frame + 1):
        for clip in ordered:
            if clip.start_frame <= index <= clip.end_frame:
                writer = writers.get(clip.index)
                if writer is None:
                    clip.path = destination / f"{clip.label()}.mp4"
                    writer = vid.VideoWriter(
                        clip.path, rate, (info.width, info.height)
                    )
                    writers[clip.index] = writer
                writer.write(frame)
            elif index > clip.end_frame and clip.index in writers:
                writers.pop(clip.index).close()
                done += 1
                if on_progress:
                    on_progress(done, len(ordered))

    for writer in writers.values():
        writer.close()
    return [c for c in ordered if c.path is not None]
