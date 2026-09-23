"""Streaming video I/O.

The original pipeline decoded an entire clip into a Python list of frames
before doing any work. That is fine for the 15-second sample it shipped with
and impossible for a real match: 90 minutes at 1080p25 is roughly 800 GB of
uncompressed frames. Everything here streams instead, so memory use is flat
regardless of how long the video is.
"""
from __future__ import annotations

import contextlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0

    @property
    def resolution(self) -> tuple[int, int]:
        return self.width, self.height


def probe(path: str | Path) -> VideoInfo:
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        # Container metadata lies often enough that a zero or absurd count has
        # to be treated as "unknown" rather than trusted.
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if count <= 0:
            count = 0
        return VideoInfo(
            path=path,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(fps) if 1.0 < fps < 240.0 else 25.0,
            frame_count=count,
        )
    finally:
        cap.release()


def frames(
    path: str | Path,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield `(frame_index, bgr_frame)` without ever holding more than one."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    try:
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = start
        while True:
            if stop is not None and idx >= stop:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - start) % stride == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()


def batches(
    path: str | Path,
    size: int = 16,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
) -> Iterator[tuple[List[int], List[np.ndarray]]]:
    """Yield `(indices, frames)` chunks, for batched model inference."""
    idxs: List[int] = []
    imgs: List[np.ndarray] = []
    for i, frame in frames(path, start=start, stop=stop, stride=stride):
        idxs.append(i)
        imgs.append(frame)
        if len(imgs) == size:
            yield idxs, imgs
            idxs, imgs = [], []
    if imgs:
        yield idxs, imgs


def sample_frames(path: str | Path, count: int = 12) -> list[tuple[int, np.ndarray]]:
    """Grab `count` frames spread evenly through the video.

    Used to bootstrap things that need a look at the match as a whole before
    processing it frame by frame — team colours, most obviously.
    """
    info = probe(path)
    total = info.frame_count
    cap = cv2.VideoCapture(str(path))
    out: list[tuple[int, np.ndarray]] = []
    try:
        if total > 0:
            targets = np.linspace(0, max(total - 1, 0), num=count, dtype=int)
            for t in targets:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
                ok, frame = cap.read()
                if ok:
                    out.append((int(t), frame))
        else:
            # Unknown length: just take the first `count` frames we can get.
            for i in range(count):
                ok, frame = cap.read()
                if not ok:
                    break
                out.append((i, frame))
    finally:
        cap.release()
    return out


class VideoWriter:
    """Writes a browser-playable MP4.

    OpenCV's H.264 support depends on how the local build was compiled, so we
    try the hardware/AVC fourccs first and fall back to mp4v, which every build
    has. `transcode_for_web` afterwards fixes up anything the browser refuses.
    """

    FOURCCS = ("avc1", "H264", "mp4v")

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = size
        self.fourcc_used: str | None = None
        self._writer: cv2.VideoWriter | None = None
        for cc in self.FOURCCS:
            writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*cc), fps, size)
            if writer.isOpened():
                self._writer = writer
                self.fourcc_used = cc
                break
            writer.release()
        if self._writer is None:
            raise IOError(f"no usable video encoder for {self.path}")

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1::-1] != self.size:
            frame = cv2.resize(frame, self.size)
        self._writer.write(frame)  # type: ignore[union-attr]

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def has_ffmpeg() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, check=True, timeout=10
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def transcode_for_web(path: str | Path) -> Path:
    """Re-encode to baseline H.264 + faststart so a browser can stream it.

    A no-op when ffmpeg is unavailable — the caller still gets a playable file,
    just one Safari or Firefox may refuse depending on the fourcc that won.
    """
    path = Path(path)
    if not has_ffmpeg():
        return path
    out = path.with_name(path.stem + "_web.mp4")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(path),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-an", str(out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
    except (OSError, subprocess.SubprocessError):
        return path
    with contextlib.suppress(OSError):
        path.unlink()
        out.rename(path)
        return path
    return out
