"""Download the open football datasets used to train the vision models.

All three are CC-BY-4.0 mirrors of the Roboflow Universe sets, already in
Ultralytics YOLO layout (images/ + labels/ + data.yaml).

    players : ball / goalkeeper / player / referee  -> detection model
    pitch   : 32 pitch landmarks                    -> pose (keypoint) model
    ball    : ball only, tight crops                -> specialist ball model
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = ROOT / "datasets"

SOURCES = {
    "players": "martinjolif/football-player-detection",
    "pitch": "martinjolif/football-pitch-detection",
    "ball": "martinjolif/football-ball-detection",
}


def fetch(name: str, repo_id: str, force: bool = False) -> Path:
    dest = DATASETS_DIR / name
    if dest.exists() and not force:
        print(f"[skip] {name} already at {dest}")
        return dest
    if dest.exists():
        shutil.rmtree(dest)

    print(f"[get ] {name}  <-  {repo_id}")
    cache = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["data/**"],
    )
    shutil.copytree(Path(cache) / "data", dest, symlinks=False)
    print(f"[ok  ] {name}  ->  {dest}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(SOURCES), nargs="*", default=sorted(SOURCES))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.only:
        fetch(name, SOURCES[name], force=args.force)


if __name__ == "__main__":
    main()
