"""Turning pixels into metres.

A broadcast camera pans, tilts and zooms constantly, so there is no single
mapping from the frame to the pitch — there is one per frame. We recover it
from the pitch's own markings: a pose model locates up to 32 known landmarks
(corners, box corners, penalty spots, centre-circle extremes), and because we
know where each of those sits on a real pitch, four or more of them determine a
homography.

Two things make this usable rather than merely correct:

  * Not every frame yields enough landmarks — a tight shot on a player may show
    none. Recent good homographies are kept and reused, so coverage gaps do not
    become holes in the analysis.
  * A homography fitted to four nearly-collinear points is numerically valid and
    physically nonsense. Every candidate is sanity-checked by reprojecting the
    pitch back into the image before it is trusted.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from typing import List, Sequence

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from . import weights
from .pitch import PITCH, PitchConfig


@dataclass
class CalibrationConfig:
    """How often to look for the pitch, and how sure to be before believing it."""

    # Landmarks move slowly relative to frame rate; solving every frame is waste.
    every_n_frames: int = 5
    # How sure the model must be about a landmark before it is used in the fit.
    # A sweep on the test split (training/scripts/evaluate.py --sweep) found no
    # meaningful difference across 0.2 to 0.5 — more landmarks make a
    # better-conditioned fit, but the extra ones are also the least accurate,
    # and the two effects cancel. Only above ~0.6 does it clearly get worse, as
    # too few points survive. Worth re-running that sweep after any further
    # training.
    keypoint_conf: float = 0.4
    min_keypoints: int = 5
    ransac_threshold: float = 8.0
    imgsz: int = 640
    device: str | None = None
    # How far back in time a homography may be reused when the current frame
    # gives us nothing, in frames.
    max_reuse_frames: int = 75
    # Blend between the solved frames on either side rather than snapping to
    # the nearer one. See CalibrationTrack.at.
    interpolate: bool = True


@dataclass
class Homography:
    frame_index: int
    matrix: np.ndarray            # image -> pitch metres, 3x3
    inverse: np.ndarray           # pitch metres -> image, 3x3
    n_points: int
    error: float                  # mean reprojection error, in pixels

    def to_pitch(self, points: np.ndarray) -> np.ndarray:
        """Image pixels -> pitch metres. Accepts (N, 2), returns (N, 2)."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.matrix).reshape(-1, 2)

    def to_image(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.inverse).reshape(-1, 2)


def _build(
    image_pts: np.ndarray,
    pitch_pts: np.ndarray,
    frame_index: int,
    frame_size: tuple[int, int],
    pitch: PitchConfig,
    ransac_threshold: float,
) -> Homography | None:
    """Fit and validate one homography. Returns None if it is not believable."""
    if len(image_pts) < 4:
        return None

    matrix, mask = cv2.findHomography(
        image_pts.astype(np.float32),
        pitch_pts.astype(np.float32),
        cv2.RANSAC,
        ransac_threshold,
    )
    if matrix is None:
        return None

    inliers = int(mask.sum()) if mask is not None else len(image_pts)
    if inliers < 4:
        return None

    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None

    # Reprojection error on the inliers: how far, in pixels, does the fitted
    # mapping put the landmarks from where the model saw them?
    used = mask.ravel().astype(bool) if mask is not None else np.ones(len(image_pts), bool)
    back = cv2.perspectiveTransform(
        pitch_pts[used].astype(np.float32).reshape(-1, 1, 2), inverse
    ).reshape(-1, 2)
    error = float(np.linalg.norm(back - image_pts[used], axis=1).mean())
    if not np.isfinite(error) or error > 60.0:
        return None

    # Degeneracy check: push the four pitch corners through the inverse and
    # require they still form a convex, non-inverted, sanely sized quad. A
    # collinear fit fails this even when RANSAC was happy.
    corners_m = np.array(
        [[0, 0], [pitch.length_m, 0], [pitch.length_m, pitch.width_m], [0, pitch.width_m]],
        dtype=np.float32,
    )
    quad = cv2.perspectiveTransform(corners_m.reshape(-1, 1, 2), inverse).reshape(-1, 2)
    if not np.all(np.isfinite(quad)):
        return None
    area = 0.5 * abs(
        np.dot(quad[:, 0], np.roll(quad[:, 1], -1)) - np.dot(quad[:, 1], np.roll(quad[:, 0], -1))
    )
    w, h = frame_size
    # The pitch is bigger than the frame in a zoomed broadcast shot, so the
    # bound is generous; it only has to reject collapse and explosion.
    if not (0.05 * w * h < area < 400.0 * w * h):
        return None

    return Homography(
        frame_index=frame_index,
        matrix=matrix,
        inverse=inverse,
        n_points=inliers,
        error=error,
    )


class PitchCalibrator:
    """Finds pitch landmarks with a pose model and turns them into homographies."""

    def __init__(self, config: CalibrationConfig | None = None, pitch: PitchConfig = PITCH):
        self.config = config or CalibrationConfig()
        self.pitch = pitch
        self.device = weights.pick_device(self.config.device)
        self.model = YOLO(str(weights.require("pitch")))
        self.vertices_m = pitch.vertices_m

    def solve_frame(self, frame_index: int, frame: np.ndarray) -> Homography | None:
        result = self.model.predict(
            frame, imgsz=self.config.imgsz, device=self.device, verbose=False
        )[0]
        kps = sv.KeyPoints.from_ultralytics(result)
        if len(kps) == 0 or kps.xy is None or kps.confidence is None:
            return None

        # The pose model emits one "pitch" instance; if it hallucinates several,
        # the most confident set of landmarks is the one to trust.
        best = int(np.argmax(kps.confidence.mean(axis=1)))
        xy = np.asarray(kps.xy[best], dtype=np.float32)
        conf = np.asarray(kps.confidence[best], dtype=np.float32)

        keep = conf >= self.config.keypoint_conf
        if keep.sum() < self.config.min_keypoints:
            return None

        h, w = frame.shape[:2]
        image_pts = xy[keep]
        pitch_pts = self.vertices_m[keep]
        # Landmarks the model places outside the frame are extrapolations, not
        # observations, and they drag the fit badly.
        inside = (
            (image_pts[:, 0] > -w * 0.1) & (image_pts[:, 0] < w * 1.1)
            & (image_pts[:, 1] > -h * 0.1) & (image_pts[:, 1] < h * 1.1)
        )
        if inside.sum() < 4:
            return None

        return _build(
            image_pts[inside], pitch_pts[inside], frame_index, (w, h),
            self.pitch, self.config.ransac_threshold,
        )


@dataclass
class CalibrationTrack:
    """The homographies found across a clip, queryable at any frame.

    Solved frames are sparse (every Nth). For a frame in between, the nearest
    solved one within `max_reuse_frames` is used — a camera does not move far in
    a fifth of a second, and a slightly stale mapping is enormously better than
    no mapping.
    """

    config: CalibrationConfig
    frames: List[int] = field(default_factory=list)
    homographies: List[Homography] = field(default_factory=list)
    # A single fixed homography supplied by hand, used when the pitch model is
    # unavailable or the footage defeats it.
    static: Homography | None = None

    # Frame index at which the current shot began. Solutions from before it
    # describe a different camera and must never be blended with what follows.
    shot_start: int = 0

    def add(self, homography: Homography | None) -> None:
        if homography is None:
            return
        self.frames.append(homography.frame_index)
        self.homographies.append(homography)

    def cut(self) -> None:
        """Declare that the camera has just changed.

        A homography is a statement about one camera looking at the pitch from
        one place. Carrying it across an edit is not approximation, it is a
        different claim entirely — and because `at()` blends the solutions
        either side of a frame, a single solution from before the cut would go
        on corrupting positions well after it. Solutions are kept, so the
        earlier shot can still be read, but everything from the old camera is
        walled off from every frame that follows.
        """
        self.shot_start = self.frames[-1] + 1 if self.frames else 0

    def at(self, frame_index: int) -> Homography | None:
        """The mapping to use for `frame_index`.

        Landmarks are solved every few frames, so most frames fall between two
        solutions. Snapping to the nearer one freezes the camera for half the
        interval and then jumps — during a pan that is exactly the frames where
        the mapping is most wrong. Blending the two instead follows the camera:
        the pitch corners are projected into the image by each neighbour,
        interpolated in time, and a fresh homography is fitted to the result.
        A pan is close enough to linear over a fifth of a second for that to
        hold.
        """
        if self.static is not None:
            return self.static
        if not self.frames:
            return None

        # Only solutions from the current shot may speak for this frame.
        if frame_index >= self.shot_start:
            first = bisect_left(self.frames, self.shot_start)
            frames, homographies = self.frames[first:], self.homographies[first:]
        else:
            frames, homographies = self.frames, self.homographies
        if not frames:
            return None

        i = bisect_left(frames, frame_index)
        before = i - 1 if 0 <= i - 1 < len(frames) else None
        after = i if 0 <= i < len(frames) else None

        # An exact hit needs no blending.
        if after is not None and frames[after] == frame_index:
            return homographies[after]

        usable = [
            j for j in (before, after)
            if j is not None and abs(frames[j] - frame_index) <= self.config.max_reuse_frames
        ]
        if not usable:
            return None
        if len(usable) == 1 or not self.config.interpolate:
            nearest = min(usable, key=lambda j: abs(frames[j] - frame_index))
            return homographies[nearest]

        a, b = usable
        blended = self._blend(homographies[a], homographies[b], frame_index)
        return blended or homographies[
            min(usable, key=lambda j: abs(frames[j] - frame_index))
        ]

    def _blend(self, first: Homography, second: Homography, frame_index: int) -> Homography | None:
        span = second.frame_index - first.frame_index
        if span <= 0:
            return None
        t = (frame_index - first.frame_index) / span
        t = min(max(t, 0.0), 1.0)

        # Four points are enough to define a homography, and the pitch corners
        # are the best-conditioned choice — spread as widely as the pitch allows.
        corners = np.array(
            [[0.0, 0.0], [PITCH.length_m, 0.0],
             [PITCH.length_m, PITCH.width_m], [0.0, PITCH.width_m]],
            dtype=np.float32,
        )
        try:
            image_a = first.to_image(corners)
            image_b = second.to_image(corners)
        except cv2.error:
            return None
        if not (np.all(np.isfinite(image_a)) and np.all(np.isfinite(image_b))):
            return None

        blended_points = (1.0 - t) * image_a + t * image_b
        matrix, _ = cv2.findHomography(blended_points.astype(np.float32), corners, 0)
        if matrix is None:
            return None
        try:
            inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return None

        return Homography(
            frame_index=frame_index,
            matrix=matrix,
            inverse=inverse,
            n_points=min(first.n_points, second.n_points),
            error=(1.0 - t) * first.error + t * second.error,
        )

    def to_pitch(self, frame_index: int, points: np.ndarray) -> np.ndarray | None:
        h = self.at(frame_index)
        return None if h is None else h.to_pitch(points)

    @property
    def coverage(self) -> float:
        """Share of solved attempts that produced a usable homography."""
        return len(self.homographies) / max(len(self.frames), 1) if self.frames else 0.0

    def summary(self) -> dict:
        if not self.homographies:
            return {"solved": 0, "mean_error_px": None, "mean_keypoints": None}
        return {
            "solved": len(self.homographies),
            "mean_error_px": round(float(np.mean([h.error for h in self.homographies])), 2),
            "mean_keypoints": round(float(np.mean([h.n_points for h in self.homographies])), 1),
        }


def manual_homography(
    image_points: Sequence[Sequence[float]],
    landmark_indices: Sequence[int],
    frame_size: tuple[int, int],
    pitch: PitchConfig = PITCH,
) -> Homography:
    """Build a homography from points a human clicked on a still frame.

    The fallback for footage the keypoint model cannot read — a phone recording
    from the stands, an unusual pitch, a heavily filtered broadcast. The caller
    supplies pixel positions and, for each, which of the 32 pitch landmarks it
    is.
    """
    image_pts = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    pitch_pts = pitch.vertices_m[np.asarray(landmark_indices, dtype=int)]
    if len(image_pts) != len(pitch_pts):
        raise ValueError("image_points and landmark_indices must be the same length")
    if len(image_pts) < 4:
        raise ValueError("at least 4 points are needed to define a homography")

    homography = _build(image_pts, pitch_pts, 0, frame_size, pitch, ransac_threshold=12.0)
    if homography is None:
        raise ValueError(
            "those points do not define a usable mapping — they may be collinear, "
            "mislabelled, or too tightly clustered"
        )
    return homography
