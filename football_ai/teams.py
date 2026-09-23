"""Deciding which team each tracked player belongs to.

The approach is unsupervised on purpose: no team, kit or league is known in
advance, so the classifier learns the two kits from the footage itself. It
samples jersey colour from frames spread across the whole clip, clusters those
samples in two, and then labels every track by majority vote over its own
appearances.

That last step is what makes it stable. Colour read from a single frame is
noisy — a player is blurred, half-occluded, standing in shadow, or turned so
that only shorts are visible. Any one reading can be wrong; a hundred readings
of the same tracked player are not.

Goalkeepers wear neither kit, so they are never clustered. They are assigned by
where they stand: a keeper belongs to the team defending the end he is in.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np
from sklearn.cluster import KMeans

HOME, AWAY = 1, 2


@dataclass
class TeamConfig:
    # The jersey lives in the upper part of a player box; the lower part is
    # shorts, socks and grass. Sampling only the torso avoids all three.
    torso_top: float = 0.12
    torso_bottom: float = 0.50
    torso_inset: float = 0.22
    min_box_height: int = 24
    # A player crop is mostly pitch, so the grass has to go before the kit
    # colour can be read. Crucially it is removed by *similarity to this
    # frame's actual grass*, not by "is it green" — kits are green often
    # enough (Wolfsburg, Celtic, Norwich) that a hue rule deletes the shirt
    # and leaves nothing to classify.
    grass_distance: float = 34.0   # CIELab distance, 0-255 axes
    min_kit_pixels: int = 24
    samples_per_frame: int = 40


def estimate_grass_colour(frame: np.ndarray) -> np.ndarray | None:
    """The colour of this pitch, in BGR, or None if no pitch is visible.

    A broad green mask is used only to *locate* the grass; the value returned
    is the median of what it found. Everything downstream compares against that
    one measured colour, so a kit that happens to be green is only discarded if
    it is genuinely the same green as the turf — floodlights, camera white
    balance and time of day all shift the pitch's colour, and this follows it.
    """
    small = cv2.resize(frame, (192, 108), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    grass = (hue >= 25) & (hue <= 100) & (sat >= 30) & (val >= 30)
    if grass.sum() < 200:
        return None
    return np.median(small[grass], axis=0)


def _lab(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3)


def _jersey_pixels(
    frame: np.ndarray,
    bbox: Sequence[float],
    cfg: TeamConfig,
    grass_bgr: np.ndarray | None = None,
) -> np.ndarray | None:
    """Kit-coloured pixels from one player box, as an (N, 3) BGR array."""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < cfg.min_box_height:
        return None

    bh, bw = y2 - y1, x2 - x1
    ty1 = y1 + int(bh * cfg.torso_top)
    ty2 = y1 + int(bh * cfg.torso_bottom)
    tx1 = x1 + int(bw * cfg.torso_inset)
    tx2 = x2 - int(bw * cfg.torso_inset)
    if ty2 - ty1 < 4 or tx2 - tx1 < 4:
        return None

    crop = frame[ty1:ty2, tx1:tx2]
    pixels = crop.reshape(-1, 3)
    if grass_bgr is None:
        return pixels if len(pixels) >= cfg.min_kit_pixels else None

    # Distance from this frame's grass, in CIELab, where equal steps look
    # roughly equally different. Nothing is filtered by brightness: a white kit
    # and a black kit are both legitimate, and clipping either would erase the
    # very difference we are trying to measure.
    lab = _lab(pixels).astype(np.float32)
    grass_lab = _lab(np.asarray(grass_bgr, dtype=np.uint8)).astype(np.float32)[0]
    distance = np.linalg.norm(lab - grass_lab, axis=1)
    kit = pixels[distance > cfg.grass_distance]
    if len(kit) < cfg.min_kit_pixels:
        # Nothing stood out from the turf — a distant player, or heavy motion
        # blur. Better to skip this sample than to classify the grass.
        return None
    return kit


def _feature(kit_pixels: np.ndarray) -> np.ndarray:
    """A 15-D descriptor of a kit: its median colour plus its hue profile.

    Median Lab alone confuses kits that differ mainly in hue at similar
    lightness (red vs. green); a hue histogram alone confuses black and white,
    which have no meaningful hue. Together they separate every pairing a match
    can present.
    """
    pixels = kit_pixels.reshape(-1, 1, 3).astype(np.uint8)
    lab = cv2.cvtColor(pixels, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV).reshape(-1, 3)

    median_lab = np.median(lab, axis=0) / 255.0

    # Weight each pixel's hue vote by its saturation: a grey pixel's hue is noise.
    weights = hsv[:, 1].astype(np.float32) / 255.0
    hist, _ = np.histogram(hsv[:, 0], bins=12, range=(0, 180), weights=weights)
    total = hist.sum()
    hist = hist / total if total > 1e-6 else np.full(12, 1.0 / 12.0)

    saturation = np.array([hsv[:, 1].mean() / 255.0])
    return np.concatenate([median_lab, hist, saturation]).astype(np.float32)


class TeamClassifier:
    """Learns the two kits, then labels tracks by vote."""

    def __init__(self, config: TeamConfig | None = None, random_state: int = 0):
        self.config = config or TeamConfig()
        self.random_state = random_state
        self.kmeans: KMeans | None = None
        self.team_colors: Dict[int, tuple[int, int, int]] = {}
        self._votes: Dict[int, Counter] = defaultdict(Counter)
        # Running mean of the kit descriptor per track. The feature is computed
        # anyway to cast the team vote; keeping it costs one addition and gives
        # the re-linking stage an appearance signal it would otherwise have to
        # decode the video a second time to obtain.
        self._kit_sums: Dict[int, np.ndarray] = {}
        self._kit_counts: Dict[int, int] = defaultdict(int)
        self._assignments: Dict[int, int] = {}
        self._fit_samples = 0

    # ---- learning the kits --------------------------------------------------

    def fit(self, samples: Iterable[tuple[np.ndarray, Sequence[Sequence[float]]]]) -> "TeamClassifier":
        """Learn the two kits from `(frame, player_boxes)` pairs.

        Give it frames from across the match, not consecutive ones: a run of
        neighbouring frames shows the same handful of players in the same light
        and teaches the clustering almost nothing.
        """
        features: List[np.ndarray] = []
        colours: List[np.ndarray] = []

        for frame, boxes in samples:
            grass = estimate_grass_colour(frame)
            for bbox in list(boxes)[: self.config.samples_per_frame]:
                kit = _jersey_pixels(frame, bbox, self.config, grass)
                if kit is None:
                    continue
                features.append(_feature(kit))
                colours.append(np.median(kit, axis=0))

        if len(features) < 8:
            raise ValueError(
                f"only {len(features)} usable jersey samples — the clip may be too "
                "short, too low-resolution, or contain no players"
            )

        X = np.stack(features)
        self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=self.random_state).fit(X)
        self._fit_samples = len(X)

        labels = self.kmeans.labels_
        palette = np.stack(colours)
        for cluster in (0, 1):
            member = palette[labels == cluster]
            bgr = np.median(member, axis=0) if len(member) else np.array([128, 128, 128])
            self.team_colors[cluster + 1] = tuple(int(round(c)) for c in bgr)

        return self

    # ---- labelling ----------------------------------------------------------

    def observe(self, frame: np.ndarray, tracks: Sequence[tuple[int, Sequence[float]]]) -> None:
        """Record one frame's worth of evidence for `(track_id, bbox)` pairs."""
        if self.kmeans is None:
            raise RuntimeError("fit() must be called before observe()")
        grass = estimate_grass_colour(frame)
        for track_id, bbox in tracks:
            kit = _jersey_pixels(frame, bbox, self.config, grass)
            if kit is None:
                continue
            feature = _feature(kit)
            cluster = int(self.kmeans.predict(feature.reshape(1, -1))[0])
            tid = int(track_id)
            self._votes[tid][cluster + 1] += 1
            if tid in self._kit_sums:
                self._kit_sums[tid] = self._kit_sums[tid] + feature
            else:
                self._kit_sums[tid] = feature.copy()
            self._kit_counts[tid] += 1

    def finalise(self) -> Dict[int, int]:
        """Resolve every track to a team by majority vote. Call once, at the end."""
        self._assignments = {
            track_id: votes.most_common(1)[0][0] for track_id, votes in self._votes.items()
        }
        return self._assignments

    def team_of(self, track_id: int) -> int | None:
        return self._assignments.get(int(track_id))

    def kit_features(self) -> Dict[int, np.ndarray]:
        """The mean kit descriptor of every track that was ever sampled.

        Averaged over the whole track, so it carries the same robustness the
        team vote does: one blurred or half-occluded reading cannot move it far.
        """
        return {
            tid: total / self._kit_counts[tid]
            for tid, total in self._kit_sums.items()
            if self._kit_counts[tid] > 0
        }

    def apply_aliases(self, mapping: Dict[int, int]) -> None:
        """Fold the evidence of merged tracks together.

        Called after re-linking has decided that several tracked fragments were
        one player. Their votes are summed rather than their conclusions being
        picked between: a fragment seen four times should not outweigh one seen
        four hundred, which is exactly what re-voting on the finalised labels
        would do. `finalise()` must be called again afterwards.
        """
        if not mapping:
            return
        votes: Dict[int, Counter] = defaultdict(Counter)
        sums: Dict[int, np.ndarray] = {}
        counts: Dict[int, int] = defaultdict(int)

        for track_id, counter in self._votes.items():
            votes[int(mapping.get(int(track_id), int(track_id)))] += counter
        for track_id, total in self._kit_sums.items():
            root = int(mapping.get(int(track_id), int(track_id)))
            sums[root] = sums[root] + total if root in sums else total.copy()
            counts[root] += self._kit_counts[track_id]

        self._votes = votes
        self._kit_sums = sums
        self._kit_counts = counts
        self._assignments = {}

    def confidence_of(self, track_id: int) -> float:
        votes = self._votes.get(int(track_id))
        if not votes:
            return 0.0
        return votes.most_common(1)[0][1] / sum(votes.values())

    # ---- goalkeepers --------------------------------------------------------

    @staticmethod
    def assign_goalkeepers(
        goalkeepers: Sequence[tuple[int, np.ndarray]],
        outfield: Sequence[tuple[int, int, np.ndarray]],
    ) -> Dict[int, int]:
        """Assign keepers to teams by which end they are standing in.

        `goalkeepers` is `(track_id, pitch_xy)`; `outfield` is
        `(track_id, team, pitch_xy)` for players already classified. A keeper
        joins whichever team's outfield players are, on average, further away
        from him along the pitch — because a keeper defends the goal his own
        team is furthest from.
        """
        if not goalkeepers or not outfield:
            return {}

        centroids: Dict[int, List[float]] = defaultdict(list)
        for _, team, xy in outfield:
            centroids[team].append(float(xy[0]))
        means = {team: float(np.mean(xs)) for team, xs in centroids.items() if xs}
        if len(means) < 2:
            return {}

        out: Dict[int, int] = {}
        for track_id, xy in goalkeepers:
            distances = {team: abs(float(xy[0]) - mean_x) for team, mean_x in means.items()}
            out[int(track_id)] = max(distances, key=distances.get)
        return out

    # ---- reporting ----------------------------------------------------------

    def summary(self) -> dict:
        return {
            "fit_samples": self._fit_samples,
            "tracks_classified": len(self._assignments),
            "mean_confidence": (
                round(
                    float(np.mean([self.confidence_of(t) for t in self._assignments])), 3
                )
                if self._assignments
                else 0.0
            ),
            "team_colors": {k: list(v) for k, v in self.team_colors.items()},
        }
