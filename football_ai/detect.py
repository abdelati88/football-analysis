"""Object detection: players, goalkeepers, referees, and the ball.

Two models cooperate here. The general detector finds everyone on the pitch,
but the ball is only a few pixels across in a wide broadcast shot and it is the
one object the whole analysis hinges on — lose the ball and you lose every
event. So a second, single-class model is run over the region the ball was last
seen in, and its answer is preferred when the general model came up empty.

That second model is trained on native-resolution 640 windows, and every path
here feeds it exactly that: a window around the last sighting, or a grid of
them across the frame when the trail has gone cold. Nothing is resized on the
way in. An earlier version trained on downscaled whole frames and inferred on
native windows, so the ball it met was three times the size of any ball it had
learned; it returned a wrong box on sixty percent of frames while appearing to
work, because it almost always returned something.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import supervision as sv
from ultralytics import YOLO

from . import weights

# Class ids as ordered in the players dataset's data.yaml.
BALL, GOALKEEPER, PLAYER, REFEREE = 0, 1, 2, 3
CLASS_NAMES = {BALL: "ball", GOALKEEPER: "goalkeeper", PLAYER: "player", REFEREE: "referee"}


def tile_windows(
    width: int, height: int, size: int, overlap: int
) -> List[tuple[int, int, int, int]]:
    """A grid of `size`-square windows covering the frame, overlapping at the seams.

    Without the overlap a ball landing on a boundary is cut in half in both
    neighbours and recognised in neither — and a boundary is exactly where an
    object is least likely to be forgiven, since half a ball looks like any
    other small bright arc.

    The last row and column are pulled back flush against the far edges rather
    than allowed to hang over them, so every window is the full size the model
    expects and no part of the frame goes unlooked at.
    """
    step = max(size - overlap, 1)
    xs = list(range(0, max(width - size, 0) + 1, step))
    ys = list(range(0, max(height - size, 0) + 1, step))
    if xs[-1] + size < width:
        xs.append(max(width - size, 0))
    if ys[-1] + size < height:
        ys.append(max(height - size, 0))
    return [
        (x, y, min(x + size, width), min(y + size, height))
        for x in xs for y in ys
    ]


@dataclass
class FrameDetections:
    """Everything the detector found in one frame."""

    index: int
    people: sv.Detections          # players, goalkeepers and referees, tracked
    ball: np.ndarray | None = None  # xyxy of the single best ball box, if any
    ball_conf: float = 0.0

    @property
    def ball_centre(self) -> np.ndarray | None:
        if self.ball is None:
            return None
        x1, y1, x2, y2 = self.ball
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


@dataclass
class DetectorConfig:
    # Match the size the model was trained at. Running inference larger than
    # training does not recover detail the weights never learned to use — it
    # costs time proportional to the area (960 is 2.2x the pixels of 640) and
    # shifts the object scales away from what the model saw, which usually
    # makes it slightly worse rather than better.
    imgsz: int = 640
    # Deliberately low. Detections below the tracker's activation threshold
    # never start a new identity, but ByteTrack's second association pass uses
    # them to keep existing ones alive — so a permissive threshold here buys
    # continuity through occlusion without buying false players.
    conf: float = 0.15
    ball_conf: float = 0.15
    iou: float = 0.5
    # A pitch holds 22 players, 3 officials and a ball. Capping detections well
    # above that keeps the weak boxes ByteTrack wants without letting a low
    # confidence threshold flood non-max suppression — which, unchecked, hits
    # its own time limit and starts silently discarding real detections.
    max_det: int = 60
    device: str | None = None
    batch: int = 8
    # Side length of the square crop searched by the specialist ball model,
    # centred on the last known ball position.
    ball_crop: int = 640
    # The whole-frame sweep is done as a grid of native-resolution windows
    # rather than one downscaled pass. Downscaling 1920 to 1280 costs a third
    # of the ball's pixels, and the ball has barely a dozen to spare; more to
    # the point, the model is trained on windows of exactly this size at
    # exactly this scale, and a resized frame is a different problem to the one
    # it learned. Eight windows cost about twice a single 1280 pass, and this
    # path only runs once the ball has been missing for several frames.
    ball_tile_overlap: int = 128
    # Whether the sweep tiles at all. It is only the right answer for a model
    # trained on native-resolution windows: measured against the earlier
    # whole-frame model, tiling made things *worse* (65% wrong boxes against
    # 22%), because a model that has never been shown an empty window answers
    # every one of the eight with a ball. Set False alongside any rollback to
    # weights trained on downscaled frames.
    ball_tiled_sweep: bool = True
    ball_sweep_imgsz: int = 1280
    # Frames without a sighting before the search widens from a crop to the
    # whole frame.
    ball_lost_after: int = 3
    use_ball_model: bool = True


class Detector:
    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self.device = weights.pick_device(self.config.device)
        self.half = weights.supports_half(self.device)

        self.model = YOLO(str(weights.require("players")))
        ball_weights = weights.resolve("ball") if self.config.use_ball_model else None
        self.ball_model = YOLO(str(ball_weights)) if ball_weights else None

        self._last_ball: np.ndarray | None = None
        self._missing = 0

    # ---- low level ----------------------------------------------------------

    def _predict(self, model: YOLO, images: Sequence[np.ndarray], conf: float, imgsz: int):
        return model.predict(
            list(images),
            imgsz=imgsz,
            conf=conf,
            iou=self.config.iou,
            max_det=self.config.max_det,
            device=self.device,
            half=self.half,
            verbose=False,
        )

    def _best_ball(self, result) -> tuple[np.ndarray | None, float]:
        det = sv.Detections.from_ultralytics(result)
        if len(det) == 0:
            return None, 0.0
        best = int(np.argmax(det.confidence))
        return det.xyxy[best].astype(np.float32).copy(), float(det.confidence[best])

    def _tiles(self, width: int, height: int) -> List[tuple[int, int, int, int]]:
        return tile_windows(
            width, height, self.config.ball_crop, self.config.ball_tile_overlap
        )

    def _ball_full_frame(self, frame: np.ndarray) -> tuple[np.ndarray | None, float]:
        """Sweep the whole frame for the ball, one native-resolution window at
        a time.

        The narrow search below only works once the ball has been seen — it
        looks near where it last was. Something has to find it the first time,
        and again after a long occlusion, or the cheap search never gets a
        chance to start.

        The windows are the size the model was trained on and are not resized,
        so the ball appears at the scale the weights actually learned. They go
        through as a single batch, which is what keeps eight passes affordable.
        """
        if self.ball_model is None:
            return None, 0.0

        h, w = frame.shape[:2]
        if not self.config.ball_tiled_sweep:
            result = self._predict(
                self.ball_model, [frame], self.config.ball_conf,
                self.config.ball_sweep_imgsz,
            )[0]
            return self._best_ball(result)
        if min(h, w) < self.config.ball_crop:
            result = self._predict(
                self.ball_model, [frame], self.config.ball_conf, self.config.ball_crop
            )[0]
            return self._best_ball(result)

        tiles = self._tiles(w, h)
        crops = [frame[y0:y1, x0:x1] for x0, y0, x1, y1 in tiles]
        results = self._predict(
            self.ball_model, crops, self.config.ball_conf, self.config.ball_crop
        )

        best_box: np.ndarray | None = None
        best_conf = 0.0
        for (x0, y0, _, _), result in zip(tiles, results):
            box, confidence = self._best_ball(result)
            if box is None or confidence <= best_conf:
                continue
            box[[0, 2]] += x0
            box[[1, 3]] += y0
            best_box, best_conf = box, confidence
        return best_box, best_conf

    def _ball_from_crop(self, frame: np.ndarray) -> tuple[np.ndarray | None, float]:
        """Second look at the ball, in a crop around where it last was.

        Cropping does two things at once: it raises the ball's effective
        resolution, and it removes most of the frame in which a false positive
        could appear.
        """
        if self.ball_model is None or self._last_ball is None:
            return None, 0.0

        h, w = frame.shape[:2]
        cx, cy = self._last_ball
        half = self.config.ball_crop // 2
        x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
        x1, y1 = int(min(w, cx + half)), int(min(h, cy + half))
        if x1 - x0 < 32 or y1 - y0 < 32:
            return None, 0.0

        crop = frame[y0:y1, x0:x1]
        result = self._predict(self.ball_model, [crop], self.config.ball_conf, self.config.ball_crop)[0]
        box, confidence = self._best_ball(result)
        if box is None:
            return None, 0.0
        box[[0, 2]] += x0
        box[[1, 3]] += y0
        return box, confidence

    # ---- public -------------------------------------------------------------

    def detect_batch(self, indices: Sequence[int], images: Sequence[np.ndarray]) -> List[FrameDetections]:
        results = self._predict(self.model, images, self.config.conf, self.config.imgsz)
        out: List[FrameDetections] = []

        for idx, frame, result in zip(indices, images, results):
            det = sv.Detections.from_ultralytics(result)

            ball_mask = det.class_id == BALL
            people = det[~ball_mask]
            # Non-max suppression across classes: a goalkeeper detected also as
            # a player would otherwise become two tracks standing on one body.
            people = people.with_nms(threshold=0.6, class_agnostic=True)

            ball_box: np.ndarray | None = None
            ball_conf = 0.0
            balls = det[ball_mask]
            if len(balls):
                best = int(np.argmax(balls.confidence))
                ball_box = balls.xyxy[best].astype(np.float32)
                ball_conf = float(balls.confidence[best])

            if ball_box is None:
                # Cheap search first, near where the ball last was; then, once
                # it has been missing long enough that "near" means nothing, a
                # full-frame sweep to reacquire it.
                ball_box, ball_conf = self._ball_from_crop(frame)
                if ball_box is None and self._missing >= self.config.ball_lost_after:
                    ball_box, ball_conf = self._ball_full_frame(frame)

            if ball_box is not None:
                self._last_ball = np.array(
                    [(ball_box[0] + ball_box[2]) / 2.0, (ball_box[1] + ball_box[3]) / 2.0],
                    dtype=np.float32,
                )
                self._missing = 0
            else:
                self._missing += 1

            out.append(FrameDetections(index=idx, people=people, ball=ball_box, ball_conf=ball_conf))

        return out

    def reset(self) -> None:
        self._last_ball = None
        self._missing = 0
