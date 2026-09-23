"""Where the trained weights live, and which device to run them on."""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.getenv("FOOTBALL_AI_MODELS", ROOT / "models"))

ModelName = Literal["players", "pitch", "ball"]

# Filenames we accept for each role, in order of preference. `best.pt` is last
# so that a freshly trained, explicitly named model always wins over whatever
# happened to be lying in models/ from an earlier experiment.
CANDIDATES: dict[str, tuple[str, ...]] = {
    "players": ("players.pt", "player_detection.pt", "best.pt"),
    "pitch": ("pitch.pt", "pitch_detection.pt", "field.pt"),
    "ball": ("ball.pt", "ball_detection.pt"),
}


def resolve(name: ModelName) -> Path | None:
    """Path to the weights for `name`, or None if they have not been trained."""
    override = os.getenv(f"FOOTBALL_AI_{name.upper()}_WEIGHTS")
    if override:
        p = Path(override)
        return p if p.exists() else None
    for filename in CANDIDATES[name]:
        p = MODELS_DIR / filename
        if p.exists():
            return p
    return None


def require(name: ModelName) -> Path:
    path = resolve(name)
    if path is None:
        looked = ", ".join(str(MODELS_DIR / c) for c in CANDIDATES[name])
        raise FileNotFoundError(
            f"no weights for '{name}'. Train them with\n"
            f"    python training/scripts/train.py {name}\n"
            f"Looked for: {looked}"
        )
    return path


def available() -> dict[str, str | None]:
    return {name: (str(p) if (p := resolve(name)) else None) for name in CANDIDATES}


@functools.lru_cache(maxsize=8)
def pick_device(requested: str | None = None) -> str:
    """cuda > mps > cpu, unless the caller insists.

    torch is imported here rather than at module scope so that asking where the
    weights are — which the web app does on every page load — does not drag in
    the whole deep-learning stack.
    """
    if requested:
        return requested
    env = os.getenv("FOOTBALL_AI_DEVICE")
    if env:
        return env

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def supports_half(device: str) -> bool:
    """Half precision is a real speedup on CUDA and a source of NaNs on MPS."""
    return device.startswith("cuda")
