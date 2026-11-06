import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_dataset import convert_line, find_splits, load_map, map_class, prepare  # noqa: E402

CMAP = load_map(ROOT / "configs" / "class_map.yaml")


def test_class_mapping_rules():
    assert map_class("Plastic", CMAP) == "plastic"
    assert map_class("BIODEGRADABLE", CMAP) == "organic"
    assert map_class("bio-degradable", CMAP) == "organic"
    assert map_class("Trash", CMAP) == "other"
    assert map_class("ALL", CMAP) is None                 # explicit drop
    assert map_class("Glass bottle", CMAP) == "glass"     # keyword token
    assert map_class("aluminium can", CMAP) == "metal"
    assert map_class("unicorn", CMAP) is None
    assert map_class("Dustbin", CMAP) == "bin"
    assert map_class("garbage bin", CMAP) == "bin"
    assert map_class("recycling bin", CMAP) == "bin"      # keyword token
    assert map_class("plastic bin", CMAP) == "bin"        # a bin made of plastic is still a bin
    assert "bin" in CMAP["target_classes"]


def test_polygon_becomes_box_and_unmapped_class_is_dropped():
    poly = "3 0.2 0.2 0.6 0.2 0.6 0.8 0.2 0.8"
    assert convert_line(poly, {3: 5}) == "5 0.400000 0.500000 0.400000 0.600000"
    assert convert_line("1 0.5 0.5 0.2 0.2", {1: 2}) == "2 0.500000 0.500000 0.200000 0.200000"
    assert convert_line("9 0.5 0.5 0.2 0.2", {1: 2}) is None
    assert convert_line("garbage", {1: 2}) is None
    assert convert_line("1 0.5 0.5 0 0.2", {1: 2}) is None          # zero-width box


def _make_export(root: Path, names, files):
    """files: {split: [(stem, [label lines])]} in Roboflow layout (train/valid/test)."""
    (root).mkdir(parents=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": names, "nc": len(names)}))
    for split, items in files.items():
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for stem, lines in items:
            (root / split / "images" / f"{stem}.jpg").write_bytes(b"img")
            (root / split / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")


def test_prepare_merges_two_datasets(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _make_export(a, ["Glass", "Plastic bottle", "ALL", "Biodegradable"], {
        "train": [("img1", ["0 0.5 0.5 0.2 0.2", "2 0.5 0.5 0.9 0.9"]),
                  ("img2", ["2 0.5 0.5 0.9 0.9"])],                      # only 'ALL' -> skipped
        "valid": [("img3", ["3 0.1 0.1 0.1 0.1"])],
    })
    _make_export(b, {0: "paper", 1: "unicorn"}, {                         # dict-style names
        "train": [("img1", ["0 0.4 0.4 0.2 0.2 ", "1 0.5 0.5 0.1 0.1"])],   # same stem as dataset a
        "valid": [("img9", ["0 0.3 0.3 0.1 0.1"])],
        "test": [("img7", ["0 0.3 0.3 0.1 0.1"])],
    })
    out = tmp_path / "out"
    data_yaml = tmp_path / "data.yaml"
    rep = prepare([a, b], out, CMAP, data_yaml=data_yaml)

    assert rep["splits"] == {"train": 2, "val": 2, "test": 1}
    assert rep["skipped_images"] == 1
    assert (out / "images" / "train" / "d0_img1.jpg").exists()
    assert (out / "images" / "train" / "d1_img1.jpg").exists()          # no name collision
    targets = CMAP["target_classes"]
    first = (out / "labels" / "train" / "d0_img1.txt").read_text().split()
    assert targets[int(first[0])] == "glass" and len(first) == 5        # 'ALL' box removed
    assert rep["instances"]["organic"] == 1 and rep["instances"]["paper"] == 3

    cfg = yaml.safe_load(data_yaml.read_text())
    assert cfg["names"][targets.index("plastic")] == "plastic"
    assert cfg["val"] == "images/val" and cfg["test"] == "images/test"


def test_dry_run_writes_nothing(tmp_path):
    a = tmp_path / "a"
    _make_export(a, ["Plastic"], {"train": [("i", ["0 0.5 0.5 0.2 0.2"])], "valid": []})
    rep = prepare([a], tmp_path / "out", CMAP, dry_run=True)
    assert rep["mapping"] == [("a", "Plastic", "plastic")]
    assert not (tmp_path / "out").exists()


def test_ultralytics_layout_is_found(tmp_path):
    for sub in ("images/train", "labels/train", "images/val", "labels/val"):
        (tmp_path / sub).mkdir(parents=True)
    assert sorted(find_splits(tmp_path)) == ["train", "val"]
