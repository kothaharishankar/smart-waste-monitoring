"""Fine-tune YOLO on your waste dataset and log the result.

Local / VS Code:   python scripts/train.py --epochs 50
Colab / Kaggle:    python scripts/train.py --data-root /content/dataset --device 0

Needs a GPU for reasonable speed (use Colab or Kaggle if your laptop has none).
"""
import argparse
import shutil
import sys
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "configs" / "data.yaml"))
    ap.add_argument("--data-root", default=str(ROOT / "data"), help="folder containing images/ and labels/")
    ap.add_argument("--model", default="yolov8s.pt", help="pretrained start (yolov8n.pt = faster, yolov8s.pt = better)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--name", default="baseline", help="experiment name, e.g. baseline, aug_v2")
    args = ap.parse_args()

    from ultralytics import YOLO

    with open(args.data, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    data_cfg["path"] = str(Path(args.data_root).resolve())
    resolved = ROOT / "configs" / "data.resolved.yaml"
    with open(resolved, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_cfg, f, sort_keys=False)

    model = YOLO(args.model)
    model.train(data=str(resolved), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, name=args.name, project=str(ROOT / "runs"), exist_ok=True)
    metrics = model.val()

    best = ROOT / "runs" / args.name / "weights" / "best.pt"
    if best.exists():
        (ROOT / "models").mkdir(exist_ok=True)
        shutil.copy(best, ROOT / "models" / "best.pt")
        print(f"Copied {best} -> models/best.pt")
    else:
        print("WARNING: best.pt not found; check the runs/ folder.", file=sys.stderr)

    # Per-material results + confusion matrix (guarded: never break training over reporting).
    try:
        lines = ["| material | precision | recall | mAP@50 |", "|---|---|---|---|"]
        for k, ci in enumerate(metrics.box.ap_class_index):
            lines.append(f"| {metrics.names[int(ci)]} | {metrics.box.p[k]:.3f} | "
                         f"{metrics.box.r[k]:.3f} | {metrics.box.ap50[k]:.3f} |")
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "results" / f"per_class_{args.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        for cm in sorted((ROOT / "runs").glob("**/confusion_matrix_normalized.png")):
            if args.name in cm.parts or cm.parent.name.startswith("val"):
                shutil.copy(cm, ROOT / "results" / f"confusion_matrix_{args.name}.png")
                break
    except Exception as e:  # noqa: BLE001
        print(f"(per-class report skipped: {e})")

    row = (f"| {date.today()} | {args.name} | {args.model} | {args.epochs} | "
           f"{metrics.box.map50:.3f} | {metrics.box.map:.3f} | {metrics.box.mp:.3f} | {metrics.box.mr:.3f} |\n")
    log = ROOT / "results" / "experiments.md"
    log.parent.mkdir(exist_ok=True)
    if not log.exists():
        log.write_text("# Experiment log\n\n| date | name | model | epochs | mAP@50 | mAP@50-95 | precision | recall |\n"
                       "|---|---|---|---|---|---|---|---|\n", encoding="utf-8")
    with open(log, "a", encoding="utf-8") as f:
        f.write(row)
    print(f"mAP@50={metrics.box.map50:.3f}  logged to results/experiments.md")


if __name__ == "__main__":
    main()
