"""The orchestrator: video in, events and statistics out.

The work is arranged as four stages so that progress can be reported honestly
to a waiting user, and so that a failure in one stage says which one it was.

  1. survey    sample frames from across the clip and learn the two kits
  2. analyse   the main pass: detect, track, calibrate, place on the pitch
  3. derive    possession, events and statistics from what was gathered
  4. render    an optional second pass that draws the result back onto the video

Nothing accumulates frames in memory. The main pass keeps flat observation
rows; the render pass re-decodes the video, which costs far less than holding
it.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import cv2
import numpy as np
import pandas as pd
import supervision as sv

from . import clips as clipper
from . import render as draw
from . import video as vid
from . import weights
from .calibrate import CalibrationConfig, CalibrationTrack, Homography, PitchCalibrator
from .detect import BALL, GOALKEEPER, PLAYER, REFEREE, Detector, DetectorConfig
from .events import Event, EventBuilder, EventConfig, infer_attacking_directions
from .pitch import PITCH, PitchConfig
from .possession import (
    PossessionConfig, Spell, assign_ball_holder, build_spells, possession_share,
)
from .segments import SegmentConfig, ShotDetector
from .stats import summarise
from .stitch import StitchConfig, StitchReport, stitch
from .teams import TeamClassifier, TeamConfig
from .track import (
    TrackConfig, TrackStore, drop_short_tracks, foot_position, interpolate_ball,
    make_tracker, smooth_positions, track_classes,
)

ProgressFn = Callable[[str, float, str], None]


def _noop(stage: str, fraction: float, message: str) -> None:
    return None


@dataclass
class PipelineConfig:
    # Analysing every frame of a broadcast buys almost nothing: possession and
    # passes resolve fine at 12-13 Hz, and halving the frames halves the run
    # time. Set to None to use the video's own rate.
    sample_fps: float | None = 12.5
    max_seconds: float | None = None
    team_names: Dict[int, str] = field(default_factory=lambda: {1: "Team 1", 2: "Team 2"})
    player_names: Dict[int, str] = field(default_factory=dict)
    half: int = 1
    render_video: bool = True
    survey_frames: int = 16

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracking: TrackConfig = field(default_factory=TrackConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    possession: PossessionConfig = field(default_factory=PossessionConfig)
    events: EventConfig = field(default_factory=EventConfig)
    teams: TeamConfig = field(default_factory=TeamConfig)
    stitch: StitchConfig = field(default_factory=StitchConfig)
    segments: SegmentConfig = field(default_factory=SegmentConfig)
    clips: clipper.ClipConfig = field(default_factory=clipper.ClipConfig)
    render: draw.RenderConfig = field(default_factory=draw.RenderConfig)
    pitch: PitchConfig = PITCH


@dataclass
class AnalysisResult:
    events: List[Event]
    stats: dict
    diagnostics: dict
    video_path: Path | None = None
    clip_paths: List[Path] = field(default_factory=list)
    duration_s: float = 0.0

    def tagger_rows(self) -> List[dict]:
        return [e.as_tagger_row() for e in self.events]


class Pipeline:
    def __init__(self, config: PipelineConfig | None = None, on_progress: ProgressFn | None = None):
        self.config = config or PipelineConfig()
        self.progress = on_progress or _noop
        self._stitch_report = StitchReport()
        self._shots: ShotDetector | None = None

    # ---- stage helpers ------------------------------------------------------

    def _report(self, stage: str, fraction: float, message: str) -> None:
        try:
            self.progress(stage, max(0.0, min(1.0, fraction)), message)
        except Exception:
            # A failing progress sink must never take the analysis down with it.
            pass

    # ---- stage 1: learn the kits -------------------------------------------

    def _survey(self, path: Path, detector: Detector) -> TeamClassifier:
        self._report("survey", 0.0, "Sampling frames from across the video")
        samples = vid.sample_frames(path, count=self.config.survey_frames)
        if not samples:
            raise RuntimeError("could not read any frames from the video")

        pairs: List[tuple[np.ndarray, List[Sequence[float]]]] = []
        for i, (index, frame) in enumerate(samples):
            det = detector.detect_batch([index], [frame])[0]
            boxes = [
                det.people.xyxy[j]
                for j in range(len(det.people))
                if det.people.class_id[j] == PLAYER
            ]
            if boxes:
                pairs.append((frame, boxes))
            self._report("survey", (i + 1) / len(samples), f"Reading kit colours ({i + 1}/{len(samples)})")

        classifier = TeamClassifier(self.config.teams)
        classifier.fit(pairs)
        detector.reset()
        self._report("survey", 1.0, "Team colours identified")
        return classifier

    # ---- stage 2: the main pass --------------------------------------------

    def _analyse(
        self,
        path: Path,
        info: vid.VideoInfo,
        detector: Detector,
        calibrator: PitchCalibrator | None,
        classifier: TeamClassifier,
        stride: int,
        stop_frame: int | None,
    ) -> tuple[TrackStore, CalibrationTrack, Dict[int, sv.Detections]]:
        store = TrackStore()
        calibration = CalibrationTrack(self.config.calibration)
        tracker = make_tracker(self.config.tracking, info.fps / stride)
        shots = ShotDetector(self.config.segments)

        total = stop_frame if stop_frame is not None else (info.frame_count or 0)
        processed = 0
        calib_every = max(self.config.calibration.every_n_frames, 1)
        sampled = 0

        for indices, images in vid.batches(
            path, size=self.config.detector.batch, stop=stop_frame, stride=stride
        ):
            detections = detector.detect_batch(indices, images)

            for det, frame in zip(detections, images):
                time_s = det.index / info.fps

                # -- shot boundary --------------------------------------------
                # Nothing carried forward survives an edit. The frame after a
                # cut has no spatial relationship to the one before it, so a
                # tracker matching by overlap will pair strangers, the ball
                # search will look where the ball cannot be, and a homography
                # solved on the previous camera will place players metres from
                # where they stand. Everything that remembers is reset here.
                if shots.observe(det.index, frame):
                    tracker = make_tracker(self.config.tracking, info.fps / stride)
                    detector.reset()
                    calibration.cut()

                # -- calibrate ------------------------------------------------
                solved = None
                if calibrator is not None and sampled % calib_every == 0:
                    solved = calibrator.solve_frame(det.index, frame)
                    calibration.add(solved)
                    shots.note_calibration(solved is not None)
                homography = calibration.at(det.index)

                # -- track ----------------------------------------------------
                tracked = tracker.update_with_detections(det.people)

                pitch_xy = None
                if homography is not None and len(tracked):
                    feet = np.stack([foot_position(b) for b in tracked.xyxy])
                    pitch_xy = homography.to_pitch(feet)

                store.add_people(det.index, time_s, tracked, pitch_xy)

                # -- team evidence --------------------------------------------
                if len(tracked) and tracked.tracker_id is not None:
                    classifier.observe(
                        frame,
                        [
                            (int(tid), tracked.xyxy[j])
                            for j, tid in enumerate(tracked.tracker_id)
                            if tid is not None and tracked.class_id[j] == PLAYER
                        ],
                    )

                # -- ball ------------------------------------------------------
                centre = det.ball_centre
                ball_pitch = (
                    homography.to_pitch(centre.reshape(1, 2))[0]
                    if (homography is not None and centre is not None)
                    else None
                )
                store.add_ball(det.index, time_s, centre, ball_pitch, det.ball_conf)

                sampled += 1
                processed = det.index

            if total:
                self._report(
                    "analyse", processed / total,
                    f"Analysing frame {processed:,} of {total:,}",
                )
            else:
                self._report("analyse", 0.5, f"Analysing frame {processed:,}")

        self._report("analyse", 1.0, f"Analysed {sampled:,} frames")
        self._shots = shots
        return store, calibration, {}

    # ---- stage 3: derive ----------------------------------------------------

    def _derive(
        self,
        store: TrackStore,
        classifier: TeamClassifier,
        info: vid.VideoInfo,
        stride: int,
    ) -> tuple[List[Event], dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, int], Dict[int, int], Dict[int, float]]:
        self._report("derive", 0.05, "Settling player identities")

        # Unfiltered on purpose: the short fragments are what the re-linking
        # below is for. Length is judged after identities have been rejoined.
        people = store.people_frame(min_track_length=0)
        ball = interpolate_ball(store.ball_frame(), max_gap=int(info.fps / stride))
        if people.empty:
            raise RuntimeError("no players were detected in this video")

        # Smooth before linking, not after: the endpoints the linking measures
        # come from these positions, and a jittering endpoint is a jittering
        # decision. Smoothing across a join would also bleed one fragment's
        # position into the gap it was supposed to be judged over.
        people = smooth_positions(people)
        classes = track_classes(people)
        teams = classifier.finalise()

        self._report("derive", 0.1, "Re-linking split identities")
        people, mapping, stitch_report = stitch(
            people,
            teams,
            classes,
            classifier.kit_features(),
            # The video's own rate, not the sampled one: `frame` holds original
            # video frame indices, so seconds must be recovered with the rate
            # those indices are numbered at. Passing the reduced rate here
            # inflates every gap by the stride, which silently pushes ordinary
            # two-second gaps past the last linking round.
            fps=info.fps,
            frame_width=float(info.width),
            config=self.config.stitch,
        )
        if stitch_report.merges:
            classifier.apply_aliases(mapping)
            teams = classifier.finalise()
            classes = track_classes(people)

        people = drop_short_tracks(people, self.config.tracking.min_track_length)
        if people.empty:
            raise RuntimeError("no players were tracked for long enough to analyse")
        self._stitch_report = stitch_report

        # Goalkeepers wear neither kit; place them by which end they occupy.
        self._report("derive", 0.15, "Assigning goalkeepers")
        placed = people[people["pitch_x"].notna()]
        if not placed.empty:
            centres = placed.groupby("track_id")[["pitch_x", "pitch_y"]].median()
            keepers = [
                (int(t), centres.loc[t].to_numpy())
                for t in centres.index
                if classes.get(int(t)) == GOALKEEPER
            ]
            outfield = [
                (int(t), teams[int(t)], centres.loc[t].to_numpy())
                for t in centres.index
                if classes.get(int(t)) == PLAYER and int(t) in teams
            ]
            teams.update(TeamClassifier.assign_goalkeepers(keepers, outfield))

        self._report("derive", 0.35, "Working out who had the ball")
        holder = assign_ball_holder(
            people, ball, teams, classes, self.config.possession, info.width
        )
        spells = build_spells(holder, people, ball, teams, classes, self.config.possession)
        possession = possession_share(holder, teams, self.config.possession)

        self._report("derive", 0.6, "Recognising events")
        directions = infer_attacking_directions(people, teams, classes, self.config.pitch)
        builder = EventBuilder(
            team_names=self.config.team_names,
            directions=directions,
            player_names=self.config.player_names,
            pitch=self.config.pitch,
            config=self.config.events,
            half=self.config.half,
            fps=info.fps,
        )
        events = builder.build(spells, ball)

        self._report("derive", 0.85, "Computing statistics")
        stats = summarise(
            events=events,
            people=people,
            holder_frame=holder,
            teams=teams,
            classes=classes,
            team_names=self.config.team_names,
            possession=possession,
            fps=info.fps,
            player_names=self.config.player_names,
            pitch=self.config.pitch,
        )
        stats["directions"] = {str(k): v for k, v in directions.items()}

        self._report("derive", 1.0, f"{len(events)} events recognised")
        return events, stats, people, ball, holder, teams, classes, possession

    # ---- stage 4: render ----------------------------------------------------

    def _render(
        self,
        path: Path,
        out_path: Path,
        info: vid.VideoInfo,
        people: pd.DataFrame,
        ball: pd.DataFrame,
        teams: Dict[int, int],
        classes: Dict[int, int],
        classifier: TeamClassifier,
        holder: pd.DataFrame,
        possession: Dict[int, float],
        stride: int,
        stop_frame: int | None,
    ) -> Path:
        self._report("render", 0.0, "Drawing the annotated video")

        by_frame = {int(f): g for f, g in people.groupby("frame")}
        ball_by_frame = {int(r.frame): r for r in ball.itertuples(index=False)}
        holder_by_frame = dict(zip(holder["frame"].astype(int), holder["holder"].astype(int)))

        colours = {
            1: classifier.team_colors.get(1, (60, 60, 220)),
            2: classifier.team_colors.get(2, (220, 120, 40)),
        }
        render_cfg = self.config.render
        radar_width = min(render_cfg.radar_width, info.width - 80)

        with vid.VideoWriter(out_path, info.fps / stride, info.resolution) as writer:
            total = stop_frame or info.frame_count or 1
            for index, frame in vid.frames(path, stop=stop_frame, stride=stride):
                group = by_frame.get(index)
                radar_players: List[tuple[np.ndarray, Sequence[int], str | None]] = []
                holder_id = holder_by_frame.get(index, -1)

                if group is not None:
                    for row in group.itertuples(index=False):
                        track_id = int(row.track_id)
                        cls = classes.get(track_id, PLAYER)
                        if cls == REFEREE:
                            colour = draw.REFEREE_COLOUR
                            label = None
                        else:
                            colour = colours.get(teams.get(track_id, 0), (170, 170, 170))
                            label = str(track_id) if render_cfg.show_track_ids else None
                        draw.draw_player(
                            frame, (row.x1, row.y1, row.x2, row.y2), colour, label,
                            has_ball=(track_id == holder_id),
                        )
                        if cls != REFEREE and not np.isnan(row.pitch_x):
                            radar_players.append(
                                (np.array([row.pitch_x, row.pitch_y]), colour, label)
                            )

                ball_row = ball_by_frame.get(index)
                ball_pitch = None
                if ball_row is not None and not np.isnan(ball_row.img_x):
                    draw.draw_ball(
                        frame, (ball_row.img_x, ball_row.img_y),
                        predicted=bool(getattr(ball_row, "interpolated", False)),
                    )
                    if not np.isnan(ball_row.pitch_x):
                        ball_pitch = np.array([ball_row.pitch_x, ball_row.pitch_y])

                if render_cfg.show_hud:
                    draw.draw_hud(
                        frame, self.config.team_names, colours, possession,
                        draw.format_clock(index / info.fps),
                    )
                if render_cfg.show_radar and radar_width > 120:
                    panel = draw.draw_radar(radar_players, ball_pitch, radar_width, self.config.pitch)
                    draw.overlay(frame, panel, render_cfg.radar_margin, render_cfg.radar_opacity)

                writer.write(frame)
                if index % 50 == 0:
                    self._report("render", index / total, f"Drawing frame {index:,} of {total:,}")

        self._report("render", 0.97, "Encoding for the browser")
        final = vid.transcode_for_web(out_path)
        self._report("render", 1.0, "Video ready")
        return final

    # ---- entry point --------------------------------------------------------

    def run(self, video_path: str | Path, output_dir: str | Path | None = None) -> AnalysisResult:
        started = time.time()
        path = Path(video_path)
        info = vid.probe(path)

        source_fps = info.fps
        target = self.config.sample_fps or source_fps
        stride = max(int(round(source_fps / target)), 1)
        stop_frame = (
            int(self.config.max_seconds * source_fps) if self.config.max_seconds else None
        )
        if stop_frame and info.frame_count:
            stop_frame = min(stop_frame, info.frame_count)

        detector = Detector(self.config.detector)
        calibrator: PitchCalibrator | None = None
        calibration_error: str | None = None
        try:
            calibrator = PitchCalibrator(self.config.calibration, self.config.pitch)
        except FileNotFoundError as exc:
            # Without the pitch model everything still runs — there are simply
            # no metres, so events cannot be placed and only image-space
            # possession survives. Say so rather than failing silently.
            calibration_error = str(exc)

        classifier = self._survey(path, detector)
        store, calibration, _ = self._analyse(
            path, info, detector, calibrator, classifier, stride, stop_frame
        )
        events, stats, people, ball, holder, teams, classes, possession = self._derive(
            store, classifier, info, stride
        )

        out_video: Path | None = None
        if self.config.render_video:
            out_dir = Path(output_dir or path.parent)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_video = self._render(
                path, out_dir / f"{path.stem}_analysed.mp4", info, people, ball,
                teams, classes, classifier, holder, possession, stride, stop_frame,
            )

        # -- clips -----------------------------------------------------------
        # A coach handed ninety minutes of annotated footage has been handed
        # ninety minutes of footage. Cutting the moments worth watching out of
        # it is what makes the analysis usable on a Monday morning.
        cut_clips: List[clipper.Clip] = []
        if self.config.clips.enabled and events:
            out_dir = Path(output_dir or path.parent) / f"{path.stem}_clips"
            self._report("render", 0.9, "Cutting the events into clips")
            planned = clipper.plan(
                [e.as_tagger_row() for e in events],
                info.fps, info.frame_count or 0, self.config.clips,
            )
            cut_clips = clipper.cut(path, planned, out_dir, on_progress=None)
            self._report("render", 1.0, f"{len(cut_clips)} clips cut")

        diagnostics = {
            "video": {
                "width": info.width, "height": info.height,
                "fps": round(info.fps, 2), "frames": info.frame_count,
                "duration_s": round(info.duration_s, 1),
            },
            "sampling": {"stride": stride, "effective_fps": round(info.fps / stride, 2)},
            "calibration": calibration.summary() | {
                "coverage": round(calibration.coverage, 3),
                "error": calibration_error,
            },
            "teams": classifier.summary(),
            "identities": self._stitch_report.as_dict(),
            "shots": self._shots.summary() if self._shots else {},
            "clips": [c.as_dict() for c in cut_clips],
            "device": weights.pick_device(self.config.detector.device),
            "weights": weights.available(),
            "runtime_s": round(time.time() - started, 1),
        }

        return AnalysisResult(
            events=events,
            stats=stats,
            diagnostics=diagnostics,
            video_path=out_video,
            clip_paths=[c.path for c in cut_clips if c.path],
            duration_s=info.duration_s,
        )


def analyse(
    video_path: str | Path,
    config: PipelineConfig | None = None,
    on_progress: ProgressFn | None = None,
    output_dir: str | Path | None = None,
) -> AnalysisResult:
    return Pipeline(config, on_progress).run(video_path, output_dir)
