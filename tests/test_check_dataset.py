import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_dataset import check, source_key  # noqa: E402


def test_source_key_groups_video_frames():
    assert source_key("video1_frame_0042") == source_key("video1_frame_0100") == "video1"
    assert source_key("bin_jpg.rf.abc123") == "bin_jpg"


def _add(root, split, stem, label="1 0.5 0.5 0.2 0.2\n"):
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "images" / split / f"{stem}.jpg").write_bytes(b"x")
    if label is not None:
        (root / "labels" / split / f"{stem}.txt").write_text(label)


def test_detects_leakage_and_missing_labels(tmp_path):
    _add(tmp_path, "train", "clipA_frame_1")
    _add(tmp_path, "val", "clipA_frame_9")           # same source in two splits
    _add(tmp_path, "train", "clipB_frame_1", label=None)
    rep = check(tmp_path, {1: "bin_overflowing"})
    assert rep["leaks"] == ["clipA"]
    assert rep["splits"]["train"]["missing_labels"] == 1
    assert rep["splits"]["train"]["classes"] == {"bin_overflowing": 1}


def test_clean_split_has_no_leaks(tmp_path):
    _add(tmp_path, "train", "clipA_frame_1")
    _add(tmp_path, "val", "clipB_frame_1")
    assert check(tmp_path)["leaks"] == []
