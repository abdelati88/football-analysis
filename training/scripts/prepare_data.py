"""Rewrite each dataset's data.yaml with absolute paths Ultralytics can resolve.

The Roboflow exports ship relative paths (`../train/images`) that only work from
one specific working directory. We normalise them once, here.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = ROOT / "datasets"

EXTRA = {
    "pitch": {
        # Landmark names, in the order the model emits them. Index -> pitch feature.
        "kpt_shape": [32, 3],
    }
}


def normalise(name: str) -> None:
    root = DATASETS_DIR / name
    cfg_path = root / "data.yaml"
    if not cfg_path.exists():
        print(f"[warn] {name}: no data.yaml, skipping")
        return

    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["path"] = str(root)
    for split, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        if (root / folder / "images").is_dir():
            cfg[split] = f"{folder}/images"
        else:
            cfg.pop(split, None)
    cfg.update(EXTRA.get(name, {}))

    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    counts = {
        s: len(list((root / f / "images").glob("*")))
        for s, f in (("train", "train"), ("val", "valid"), ("test", "test"))
        if (root / f / "images").is_dir()
    }
    print(f"[ok  ] {name}: {counts}")


if __name__ == "__main__":
    for ds in ("players", "pitch", "ball"):
        normalise(ds)
