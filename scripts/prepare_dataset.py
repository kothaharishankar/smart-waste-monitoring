"""Merge one or more downloaded YOLO datasets into data/ with UNIFIED material classes.

Downloaded datasets use their own class names ("Plastic bottle", "BIODEGRADABLE", "Glass"...).
This script maps them onto the classes in configs/class_map.yaml, converts polygon labels to
boxes, renames files so datasets cannot collide, and writes configs/data.yaml.

    # 1) look at how classes would be mapped - nothing is written
    python scripts/prepare_dataset.py --src "C:\\path\\to\\dataset1" --src "C:\\path\\to\\dataset2" --dry-run

    # 2) build data/images + data/labels
    python scripts/prepare_dataset.py --src "C:\\path\\to\\dataset1" --src "C:\\path\\to\\dataset2" --clean

Supported layouts (Roboflow "YOLOv8" export, or Ultralytics style):
    dataset/data.yaml  dataset/train/images  dataset/train/labels  dataset/valid/...  dataset/test/...
    dataset/data.yaml  dataset/images/train  dataset/labels/train  ...

COCO-format downloads (Roboflow "COCO" export, TACO, GBS, Hugging Face ...) are detected
automatically when the folder has no data.yaml but contains COCO .json file(s):
    dataset/train/_annotations.coco.json  (+ images next to it)     -> split from folder/file name
    dataset/annotations.json  + dataset/images/...                  -> no split name: 80/10/10 by file name
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zlib
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}


def load_map(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]


def map_class(name: str, cmap: dict):
    """Source class name -> target material, or None to drop it."""
    key = name.strip().lower()
    if key in {d.lower() for d in cmap.get("drop", [])}:
        return None
    explicit = {k.lower(): v for k, v in cmap.get("explicit", {}).items()}
    if key in explicit:
        return explicit[key]
    if key in cmap["target_classes"]:
        return key
    toks = set(_tokens(name))
    for material, words in cmap.get("keywords", {}).items():
        if any(w in toks or w == key for w in words):
            return material
    return None


def read_names(src: Path) -> list[str]:
    ypath = src / "data.yaml"
    if not ypath.exists():
        raise FileNotFoundError(f"{src}: no data.yaml found - is this a YOLO-format export?")
    names = yaml.safe_load(open(ypath, encoding="utf-8")).get("names")
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=int)]
    return list(names)


def find_splits(src: Path) -> dict:
    found: dict = {}
    for alias, split in SPLIT_ALIASES.items():
        for img, lbl in ((src / alias / "images", src / alias / "labels"),
                         (src / "images" / alias, src / "labels" / alias)):
            if img.is_dir():
                found.setdefault(split, (img, lbl))
                break
    return found


def convert_line(line: str, id_map: dict):
    """One YOLO label line -> remapped box line, or None. Polygons become boxes."""
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        new_id = id_map.get(int(float(parts[0])))
        vals = [float(v) for v in parts[1:]]
    except ValueError:
        return None
    if new_id is None:
        return None
    if len(vals) == 4:
        x, y, w, h = vals
    elif len(vals) >= 6 and len(vals) % 2 == 0:          # polygon: x1 y1 x2 y2 ...
        xs, ys = vals[0::2], vals[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    else:
        return None
    x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
    w, h = min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0)
    if w <= 0 or h <= 0:
        return None
    return f"{new_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


# ----------------------------------------------------------------------------- COCO support
def _is_coco(path: Path) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            head = f.read(2000)
        return '"images"' in head or '"annotations"' in head or '"categories"' in head
    except (OSError, UnicodeDecodeError):
        return False


def find_coco_jsons(src: Path) -> list:
    """Every COCO annotation file inside src (full check on the few that look like COCO)."""
    out = []
    for p in sorted(src.rglob("*.json")):
        if not _is_coco(p):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and {"images", "annotations", "categories"} <= set(d):
            out.append((p, d))
    return out


def _split_from_name(path: Path, root: Path):
    """'train' / 'val' / 'test' if the file or one of its folders says so, else None."""
    rel = path.relative_to(root) if root in path.parents else path
    for part in [path.stem.lower(), *[q.lower() for q in rel.parts[:-1]]]:
        for tok in _tokens(part):
            if tok in SPLIT_ALIASES:
                return SPLIT_ALIASES[tok]
    return None


def _auto_split(key: str) -> str:
    h = zlib.crc32(key.encode("utf-8")) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def coco_names(src: Path) -> dict:
    """category id -> name, merged over all COCO files of a source."""
    names: dict = {}
    for _, d in find_coco_jsons(src):
        for c in d["categories"]:
            names[int(c["id"])] = str(c["name"])
    if not names:
        raise FileNotFoundError(f"{src}: no data.yaml and no COCO .json annotation file found")
    return names


def coco_entries(src: Path):
    """Yield (split, image_path, image_id_tag, [(category_id, x, y, w, h normalised)]) per image."""
    index = None
    for jpath, d in find_coco_jsons(src):
        split_of_file = _split_from_name(jpath, src)
        anns: dict = {}
        for a in d["annotations"]:
            if a.get("iscrowd"):
                continue
            anns.setdefault(a["image_id"], []).append(a)
        for im in d["images"]:
            fname = im["file_name"].replace("\\", "/")
            cand = [jpath.parent / fname, jpath.parent.parent / fname, src / fname,
                    jpath.parent / "images" / Path(fname).name]
            img = next((c for c in cand if c.is_file()), None)
            if img is None:
                if index is None:                       # last resort: find by file name anywhere
                    index = {}
                    for q in src.rglob("*"):
                        if q.suffix.lower() in IMG_EXT:
                            index.setdefault(q.name, q)
                img = index.get(Path(fname).name)
            if img is None:
                continue
            W, H = float(im.get("width") or 0), float(im.get("height") or 0)
            if W <= 0 or H <= 0:
                continue
            boxes = []
            for a in anns.get(im["id"], []):
                bx, by, bw, bh = (float(v) for v in a["bbox"])
                boxes.append((int(a["category_id"]), (bx + bw / 2) / W, (by + bh / 2) / H, bw / W, bh / H))
            split = split_of_file or _auto_split(fname)
            yield split, img, f"{im['id']}", boxes


def prepare(srcs, out: Path, cmap: dict, dry_run: bool = False, clean: bool = False,
            data_yaml: Path | None = None) -> dict:
    targets = list(cmap["target_classes"])
    tindex = {t: i for i, t in enumerate(targets)}
    report = {"mapping": [], "splits": {}, "instances": Counter(), "skipped_images": 0}

    if clean and not dry_run:
        for sub in ("images", "labels"):
            shutil.rmtree(out / sub, ignore_errors=True)

    for si, src in enumerate(map(Path, srcs)):
        coco = not (src / "data.yaml").exists() and src.is_dir()
        if coco:
            names_by_id = coco_names(src)
            id_keys = sorted(names_by_id)
            names = [names_by_id[k] for k in id_keys]
        else:
            names = read_names(src)
            id_keys = list(range(len(names)))
        id_map = {}
        for key, name in zip(id_keys, names):
            tgt = map_class(name, cmap)
            report["mapping"].append((src.name, name, tgt))
            if tgt is not None:
                if tgt not in tindex:
                    raise ValueError(f"class_map.yaml maps '{name}' to '{tgt}', which is not in target_classes")
                id_map[key] = tindex[tgt]
        if dry_run:
            continue

        def save(split, img, stem, lines):
            if not lines:
                report["skipped_images"] += 1
                return
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, out / "images" / split / f"{stem}{img.suffix.lower()}")
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            report["instances"].update(targets[int(l.split()[0])] for l in lines)
            report["splits"][split] = report["splits"].get(split, 0) + 1

        if coco:
            for split, img, tag, boxes in coco_entries(src):
                lines = [c for c in (convert_line(f"{cid} {x} {y} {w} {h}", id_map)
                                     for cid, x, y, w, h in boxes) if c]
                save(split, img, f"d{si}_{tag}_{img.stem}", lines)
            continue

        splits = find_splits(src)
        if "train" not in splits or "val" not in splits:
            raise FileNotFoundError(f"{src}: need both a train and a val/valid folder (found {sorted(splits)})")
        for split, (img_dir, lbl_dir) in splits.items():
            report["splits"].setdefault(split, 0)
            for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT):
                lbl = lbl_dir / f"{img.stem}.txt"
                lines = []
                if lbl.exists():
                    lines = [c for c in (convert_line(l, id_map) for l in lbl.read_text().splitlines()) if c]
                save(split, img, f"d{si}_{img.stem}", lines)

    if not dry_run:
        yaml_path = data_yaml or (ROOT / "configs" / "data.yaml")
        cfg = {"path": "data", "train": "images/train", "val": "images/val",
               "names": {i: n for i, n in enumerate(targets)}}
        if report["splits"].get("test"):
            cfg["test"] = "images/test"
        yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        report["data_yaml"] = str(yaml_path)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", action="append", required=True, help="a downloaded dataset folder, YOLO or COCO format (repeatable)")
    ap.add_argument("--out", default=str(ROOT / "data"))
    ap.add_argument("--map", default=str(ROOT / "configs" / "class_map.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="only print the class mapping")
    ap.add_argument("--clean", action="store_true", help="delete existing data/images and data/labels first")
    args = ap.parse_args(argv)

    cmap = load_map(args.map)
    rep = prepare(args.src, Path(args.out), cmap, dry_run=args.dry_run, clean=args.clean)

    print("\nClass mapping (CHECK THIS - edit configs/class_map.yaml if something is wrong):")
    for ds, src_name, tgt in rep["mapping"]:
        print(f"  [{ds}] {src_name!r:<28} -> {tgt if tgt else 'DROPPED'}")
    dropped = sorted({n for _, n, t in rep["mapping"] if t is None})
    if dropped:
        print(f"\nDropped (no mapping): {dropped}")
    if args.dry_run:
        print("\nDry run - nothing written.")
        return
    print("\nImages kept per split:", rep["splits"], f"| images skipped (no usable boxes): {rep['skipped_images']}")
    print("Instances per material:")
    for k, v in sorted(rep["instances"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:<10} {v}")
    print(f"\nWrote {rep['data_yaml']}. Next: python scripts/check_dataset.py --root data")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"ERROR: {e}")