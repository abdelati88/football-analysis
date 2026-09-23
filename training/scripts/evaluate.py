"""Measure how good the trained models actually are.

Produces the numbers a write-up needs: detection accuracy per class, keypoint
accuracy, and — the one that matters most for this project and that no standard
metric reports — how well the pitch keypoints translate into a homography, in
metres of error on the ground.

    python training/scripts/evaluate.py                 # every trained model
    python training/scripts/evaluate.py players --split test
    python training/scripts/evaluate.py --markdown report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from football_ai import weights                      # noqa: E402
from football_ai.calibrate import _build             # noqa: E402
from football_ai.pitch import PITCH                  # noqa: E402

DATASETS = ROOT / "datasets"


def evaluate_detection(name: str, split: str) -> dict[str, Any] | None:
    """Standard detection metrics, plus a per-class breakdown.

    The per-class numbers are the interesting part: mean average precision over
    all four classes hides the fact that players are easy and the ball is not.
    """
    from ultralytics import YOLO

    path = weights.resolve(name)
    if path is None:
        return None

    model = YOLO(str(path))
    results = model.val(
        data=str(DATASETS / ("players" if name == "players" else name) / "data.yaml"),
        split=split, verbose=False, plots=False, device=weights.pick_device(),
    )
    box = results.box
    per_class = {}
    for i, class_index in enumerate(box.ap_class_index):
        per_class[results.names[int(class_index)]] = {
            "precision": round(float(box.p[i]), 4),
            "recall": round(float(box.r[i]), 4),
            "mAP50": round(float(box.ap50[i]), 4),
            "mAP50_95": round(float(box.ap[i]), 4),
        }
    return {
        "model": name,
        "weights": str(path),
        "split": split,
        "overall": {
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "mAP50": round(float(box.map50), 4),
            "mAP50_95": round(float(box.map), 4),
        },
        "per_class": per_class,
    }


def evaluate_pose(split: str) -> dict[str, Any] | None:
    from ultralytics import YOLO

    path = weights.resolve("pitch")
    if path is None:
        return None
    model = YOLO(str(path))
    results = model.val(
        data=str(DATASETS / "pitch" / "data.yaml"),
        split=split, verbose=False, plots=False, device="cpu",
    )
    return {
        "model": "pitch",
        "weights": str(path),
        "split": split,
        "box": {
            "precision": round(float(results.box.mp), 4),
            "recall": round(float(results.box.mr), 4),
            "mAP50": round(float(results.box.map50), 4),
        },
        "keypoints": {
            "precision": round(float(results.pose.mp), 4),
            "recall": round(float(results.pose.mr), 4),
            "mAP50": round(float(results.pose.map50), 4),
            "mAP50_95": round(float(results.pose.map), 4),
        },
    }


def evaluate_homography(split: str, conf: float = 0.5) -> dict[str, Any] | None:
    """The metric this project actually depends on.

    Keypoint mAP says how often a landmark lands within a tolerance. It does not
    say whether the homography built from those landmarks puts a *player* in the
    right place — which is what every distance, every event position and the
    whole radar rest on.

    So this is a hold-out test. The pitch's landmarks are split in two by index,
    fixed and deterministic: even-numbered landmarks may be used to fit the
    homography, odd-numbered ones only ever to score it. The error reported is
    how far, in metres, the fitted mapping puts the odd landmarks from where
    they truly are.

    The split has to be fixed rather than "whatever the model was not confident
    about", because that would shrink the test set every time the confidence
    threshold is lowered — making a laxer threshold look better purely because
    it was marked on an easier paper.
    """
    import cv2
    from ultralytics import YOLO
    import supervision as sv

    path = weights.resolve("pitch")
    if path is None:
        return None

    images_dir = DATASETS / "pitch" / ("valid" if split == "val" else split) / "images"
    labels_dir = DATASETS / "pitch" / ("valid" if split == "val" else split) / "labels"
    if not images_dir.is_dir():
        return None

    model = YOLO(str(path))
    vertices = PITCH.vertices_m
    errors: list[float] = []
    fit_errors: list[float] = []
    used: list[int] = []
    solved = 0
    total = 0

    for image_path in sorted(images_dir.glob("*.jpg")):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        total += 1

        frame = cv2.imread(str(image_path))
        h, w = frame.shape[:2]

        # Ground truth: YOLO pose labels are cls cx cy bw bh then x y v triples,
        # all normalised.
        parts = label_path.read_text().split()
        if len(parts) < 5 + 32 * 3:
            continue
        kp = np.array(parts[5:5 + 32 * 3], dtype=np.float32).reshape(32, 3)
        truth_xy = kp[:, :2] * np.array([w, h], dtype=np.float32)
        truth_visible = kp[:, 2] > 0

        result = model.predict(frame, imgsz=640, device="cpu", verbose=False)[0]
        kps = sv.KeyPoints.from_ultralytics(result)
        if len(kps) == 0 or kps.confidence is None:
            continue
        best = int(np.argmax(kps.confidence.mean(axis=1)))
        pred_xy = np.asarray(kps.xy[best], dtype=np.float32)
        pred_conf = np.asarray(kps.confidence[best], dtype=np.float32)

        # The fixed split: even landmarks may be fitted, odd ones only scored.
        may_fit = np.zeros(32, dtype=bool)
        may_fit[::2] = True

        keep = (pred_conf >= conf) & may_fit
        if keep.sum() < 4:
            continue
        homography = _build(pred_xy[keep], vertices[keep], 0, (w, h), PITCH, 8.0)
        if homography is None:
            continue
        solved += 1
        used.append(int(keep.sum()))

        # How well it fits what it was given — optimistically biased, and
        # reported only as a contrast to the honest number below.
        placed = homography.to_pitch(truth_xy[keep])
        fit_errors.append(float(np.linalg.norm(placed - vertices[keep], axis=1).mean()))

        # The honest number: landmarks this homography has never seen.
        held = truth_visible & ~may_fit
        if held.sum() >= 3:
            placed = homography.to_pitch(truth_xy[held])
            errors.append(float(np.linalg.norm(placed - vertices[held], axis=1).mean()))

    if not errors:
        return {"model": "homography", "split": split, "images": total, "solved": solved,
                "note": "no image produced a usable homography"}

    errors_array = np.array(errors)
    fit_array = np.array(fit_errors) if fit_errors else np.array([np.nan])
    return {
        "model": "homography",
        "split": split,
        "images": total,
        "solved": solved,
        "solve_rate": round(solved / max(total, 1), 4),
        "keypoint_confidence": conf,
        "landmarks_used": round(float(np.mean(used)), 1) if used else 0.0,
        "fit_error_m": round(float(np.nanmean(fit_array)), 3),
        "mean_error_m": round(float(errors_array.mean()), 3),
        "median_error_m": round(float(np.median(errors_array)), 3),
        "p90_error_m": round(float(np.percentile(errors_array, 90)), 3),
        "within_1m": round(float((errors_array <= 1.0).mean()), 4),
        "within_2m": round(float((errors_array <= 2.0).mean()), 4),
        "within_5m": round(float((errors_array <= 5.0).mean()), 4),
    }


def to_markdown(report: list[dict]) -> str:
    lines = ["# Model evaluation", ""]
    for section in report:
        if section is None:
            continue
        name = section["model"]
        lines.append(f"## {name}  ({section.get('split', '')})")
        lines.append("")
        if name == "homography":
            lines += [
                "| metric | value |", "|---|---|",
                f"| images | {section.get('images')} |",
                f"| homography solved | {section.get('solved')} ({section.get('solve_rate', 0) * 100:.0f}%) |",
                f"| error at fitted landmarks | {section.get('fit_error_m')} m |",
                f"| error at held-out landmarks | {section.get('mean_error_m')} m |",
                f"| median error | {section.get('median_error_m')} m |",
                f"| 90th percentile | {section.get('p90_error_m')} m |",
                f"| within 1 m | {section.get('within_1m', 0) * 100:.0f}% |",
                f"| within 2 m | {section.get('within_2m', 0) * 100:.0f}% |",
                "",
            ]
        elif "per_class" in section:
            o = section["overall"]
            lines += [
                f"Overall: precision {o['precision']}, recall {o['recall']}, "
                f"mAP50 {o['mAP50']}, mAP50-95 {o['mAP50_95']}", "",
                "| class | precision | recall | mAP50 | mAP50-95 |", "|---|---|---|---|---|",
            ]
            for cls, m in section["per_class"].items():
                lines.append(f"| {cls} | {m['precision']} | {m['recall']} | {m['mAP50']} | {m['mAP50_95']} |")
            lines.append("")
        else:
            k = section["keypoints"]
            lines += [
                "| metric | box | keypoints |", "|---|---|---|",
                f"| precision | {section['box']['precision']} | {k['precision']} |",
                f"| recall | {section['box']['recall']} | {k['recall']} |",
                f"| mAP50 | {section['box']['mAP50']} | {k['mAP50']} |",
                "",
            ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    # `choices` is checked manually below rather than passed to argparse:
    # with nargs="*" argparse validates the whole *list* against choices, so
    # both a list default and an empty list are rejected as invalid values.
    ap.add_argument("models", nargs="*", metavar="MODEL",
                    help="players, ball, pitch or homography (default: all)")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--json", type=Path)
    ap.add_argument("--markdown", type=Path)
    ap.add_argument("--keypoint-conf", type=float, default=0.5, dest="keypoint_conf",
                    help="confidence a landmark needs before it is used in the fit")
    ap.add_argument("--sweep", action="store_true",
                    help="try several keypoint confidences and compare them")
    args = ap.parse_args()

    known = ["players", "ball", "pitch", "homography"]
    models = args.models or known
    unknown = [m for m in models if m not in known]
    if unknown:
        print(f"Unknown model(s): {', '.join(unknown)}. Choose from: {', '.join(known)}",
              file=sys.stderr)
        return 1

    if args.sweep:
        print("\n  keypoint    solved   landmarks   held-out error   within 5 m")
        print("  confidence            used        (mean, metres)")
        for conf in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            section = evaluate_homography(args.split, conf=conf)
            if not section or "mean_error_m" not in section:
                print(f"  {conf:<11.2f} — not enough landmarks to score")
                continue
            print(
                f"  {conf:<11.2f} {section['solve_rate'] * 100:5.0f}%   "
                f"{section['landmarks_used']:>6.1f}      {section['mean_error_m']:>8.2f}      "
                f"{section['within_5m'] * 100:>6.0f}%"
            )
        print(
            "\n  Only 28 test images, so the mean is noisy — read the last column.\n"
            "  Re-run this after any further training; the best threshold moves\n"
            "  with the model's own confidence calibration.\n"
        )
        return 0

    report: list[dict] = []
    for name in models:
        print(f"\n--- evaluating {name} ---")
        if name in ("players", "ball"):
            section = evaluate_detection(name, args.split)
        elif name == "pitch":
            section = evaluate_pose(args.split)
        else:
            section = evaluate_homography(args.split, conf=args.keypoint_conf)

        if section is None:
            print(f"  skipped: no weights for '{name}'")
            continue
        report.append(section)
        print(json.dumps(section, indent=2))

    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nWrote {args.json}")
    if args.markdown:
        args.markdown.write_text(to_markdown(report))
        print(f"Wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
