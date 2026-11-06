"""Sanity-check a YOLO dataset before training.

    python scripts/check_dataset.py --root data

Reports image/label counts per split, class balance, images without labels,
and warns if frames from the same source video/photo set appear in more than
one split (data leakage inflates your mAP).
"""
import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def source_key(stem: str) -> str:
    """Best-effort 'where did this image come from': strips Roboflow's '.rf.<hash>'
    suffix and trailing frame numbers (video1_frame_0042 -> video1)."""
    key = stem.split(".rf.")[0]
    key = re.sub(r"[_-]?(frame|img|image)?[_-]?\d+$", "", key, flags=re.I)
    return key or stem


def check(root: Path, names: dict | None = None) -> dict:
    report = {"splits": {}, "leaks": []}
    keys_by_split: dict[str, set] = defaultdict(set)
    for split in ("train", "val", "test"):
        img_dir, lbl_dir = root / "images" / split, root / "labels" / split
        imgs = [p for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXT] if img_dir.exists() else []
        missing, classes = 0, Counter()
        for p in imgs:
            keys_by_split[split].add(source_key(p.stem))
            lbl = lbl_dir / f"{p.stem}.txt"
            if not lbl.exists():
                missing += 1
                continue
            for line in lbl.read_text().splitlines():
                if line.strip():
                    cid = int(float(line.split()[0]))
                    classes[(names or {}).get(cid, cid)] += 1
        report["splits"][split] = {"images": len(imgs), "missing_labels": missing, "classes": dict(classes)}
    splits = list(keys_by_split)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            report["leaks"] += sorted(keys_by_split[a] & keys_by_split[b])
    report["leaks"] = sorted(set(report["leaks"]))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--data-yaml", default="configs/data.yaml")
    args = ap.parse_args()
    names = None
    if Path(args.data_yaml).exists():
        names = yaml.safe_load(open(args.data_yaml, encoding="utf-8")).get("names")
    rep = check(Path(args.root), names)
    for split, s in rep["splits"].items():
        print(f"{split:<5} images={s['images']:<6} missing_labels={s['missing_labels']:<4} classes={s['classes']}")
    if rep["leaks"]:
        print(f"\nWARNING: possible leakage - these sources appear in several splits: {rep['leaks'][:10]}")
        print("Split by camera/location/video instead of randomly.")
    else:
        print("\nNo cross-split source overlap detected (by filename heuristic).")


if __name__ == "__main__":
    main()
