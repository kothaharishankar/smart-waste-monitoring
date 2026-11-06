"""See the waste segregation on your own images.

    python scripts/predict_image.py --source path/to/photo.jpg
    python scripts/predict_image.py --source path/to/folder --out results/predictions

For every image it saves an annotated copy (each material has its own colour) and prints
how many items of each material were found and their share of the waste area.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve  # noqa: E402
from src.segregate import analyse_image, shares  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="an image file or a folder of images")
    ap.add_argument("--out", default="results/predictions")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    from src.detector import Detector

    cfg = load_config(args.config)
    m = cfg["model"]
    try:
        detector = Detector(resolve(args.weights or m["weights"]), m["conf"], m["imgsz"], m.get("device"))
    except FileNotFoundError as e:
        sys.exit(f"ERROR: {e}")

    src = Path(args.source)
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not files:
        sys.exit(f"No images found in {src}")
    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_counts, total_area = Counter(), Counter()
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"skip (unreadable): {f}")
            continue
        annotated, info = analyse_image(detector, img, cfg)
        cv2.imwrite(str(out_dir / f"{f.stem}_segregated.jpg"), annotated)
        total_counts.update(info.counts)
        total_area.update(info.comp)
        mix = ", ".join(f"{k} {v:.0%}" for k, v in shares(info.comp).items()) or "-"
        bins = f"bins={len(info.bins)} overflowing={sum(b['overflowing'] for b in info.bins)}"
        verdict = info.label if info.flagged else "ok"
        print(f"{f.name:<30} items={info.count:<3} {bins:<26} alert={verdict:<16} mix: {mix}")

    print("\nTotal items per material:")
    for k, v in total_counts.most_common():
        print(f"  {k:<10} {v}")
    print(f"\nAnnotated images saved to {out_dir}")


if __name__ == "__main__":
    main()
