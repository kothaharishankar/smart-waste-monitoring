"""End-to-end pipeline test with a FAKE detector and a synthetic video, so it runs
without a trained model. Bright frames = 'waste present' (2 plastic + 1 metal item)."""
import json

import cv2
import numpy as np
import pytest

from src import db
from src.config import load_config
from src.detector import Detection, draw
from src.events import raw_segments
from src.pipeline import process_video

FPS, W, H = 10, 160, 120


class FakeDetector:
    """Bright frame -> three waste items; dark frame -> nothing."""
    def detect(self, frame):
        if frame.mean() > 100:
            return [Detection("plastic", 0.9, (5, 5, 45, 45)),
                    Detection("plastic", 0.9, (55, 5, 95, 45)),
                    Detection("metal", 0.9, (5, 55, 45, 95))]
        return []


class FakeBinDetector:
    """Bright frame -> a bin with two plastic items above its rim and a glass item inside it."""
    def detect(self, frame):
        if frame.mean() <= 100:
            return [Detection("bin", 0.9, (60, 40, 100, 100))]
        return [Detection("bin", 0.9, (60, 40, 100, 100)),
                Detection("plastic", 0.9, (65, 20, 80, 34)),
                Detection("plastic", 0.9, (80, 22, 95, 36)),
                Detection("glass", 0.9, (70, 60, 80, 70))]


class FakeDirectDetector:
    """Bright frame -> one 'bin_overflowing' box (the optional direct class)."""
    def detect(self, frame):
        return [Detection("bin_overflowing", 0.9, (10, 10, 120, 100))] if frame.mean() > 100 else []


def make_video(path, bright_seconds, total_s=60):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for i in range(total_s * FPS):
        v = 200 if (i // FPS) in bright_seconds else 20
        w.write(np.full((H, W, 3), v, np.uint8))
    w.release()


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c["events"].update(window=5, on_ratio=0.6, off_ratio=0.2)
    c["paths"]["alerts_dir"] = str(tmp_path / "alerts")
    return c


def test_sustained_pile_becomes_one_logged_event_with_material_mix(tmp_path, cfg):
    video = tmp_path / "incident.avi"
    make_video(video, bright_seconds=range(10, 40))
    conn = db.connect(tmp_path / "t.db")
    for c in cfg["cameras"]:
        db.upsert_camera(conn, c)
    res = process_video(video, FakeDetector(), cfg, conn=conn, alerts_dir=tmp_path / "alerts")
    conn.close()

    assert len(res.events) == 1
    ev = res.events[0]
    assert ev["type"] == "garbage_dump"
    assert 9 <= ev["start_s"] <= 11 and 38 <= ev["end_s"] <= 40
    assert ev["composition"] == {"plastic": 0.667, "metal": 0.333}
    assert any((tmp_path / "alerts").iterdir())

    df = db.fetch_events(tmp_path / "t.db")
    assert len(df) == 1 and df.loc[0, "end_ts"] is not None and df.loc[0, "camera_name"]
    assert json.loads(df.loc[0, "composition"]) == {"plastic": 0.667, "metal": 0.333}
    assert 55 <= res.frames_sampled <= 61


def test_flicker_produces_no_alert_but_raw_baseline_spams(tmp_path, cfg):
    video = tmp_path / "flicker.avi"
    make_video(video, bright_seconds={5, 15, 25, 35})
    res = process_video(video, FakeDetector(), cfg, alerts_dir=tmp_path / "alerts")
    assert res.events == []
    assert len(raw_segments(res.trace)) == 4


def test_overflowing_bin_event_blames_only_the_waste_outside_the_bin(tmp_path, cfg):
    video = tmp_path / "bin.avi"
    make_video(video, bright_seconds=range(10, 40))
    res = process_video(video, FakeBinDetector(), cfg, alerts_dir=tmp_path / "alerts")
    assert len(res.events) == 1
    assert res.events[0]["type"] == "bin_overflowing"
    assert res.events[0]["composition"] == {"plastic": 1.0}      # the glass inside the bin is not counted


def test_bin_with_waste_only_inside_never_alerts(tmp_path, cfg):
    video = tmp_path / "quiet.avi"
    make_video(video, bright_seconds=set(), total_s=40)
    res = process_video(video, FakeBinDetector(), cfg, alerts_dir=tmp_path / "alerts")
    assert res.events == []


def test_direct_class_event_type_is_used(tmp_path, cfg):
    video = tmp_path / "direct.avi"
    make_video(video, bright_seconds=range(5, 30), total_s=40)
    res = process_video(video, FakeDirectDetector(), cfg, alerts_dir=tmp_path / "alerts")
    assert len(res.events) == 1 and res.events[0]["type"] == "bin_overflowing"
    assert res.events[0]["composition"] == {}          # no material items, only the direct class


def test_roi_can_hide_the_waste(tmp_path, cfg):
    video = tmp_path / "roi.avi"
    make_video(video, bright_seconds=range(10, 40))
    cam = dict(cfg["cameras"][0], roi=[0.9, 0.9, 1.0, 1.0])        # corner with no waste in it
    res = process_video(video, FakeDetector(), cfg, camera=cam, alerts_dir=tmp_path / "alerts")
    assert res.events == []


def test_missing_source_raises(cfg, tmp_path):
    with pytest.raises(RuntimeError):
        process_video(tmp_path / "nope.mp4", FakeDetector(), cfg)


def test_draw_keeps_shape_and_uses_colours():
    frame = np.zeros((H, W, 3), np.uint8)
    out = draw(frame, [Detection("plastic", 0.9, (1, 1, 50, 50))], {"garbage_dump"}, "x")
    assert out.shape == frame.shape and out is not frame and out.any()


def test_fetch_events_on_missing_db_is_empty(tmp_path):
    df = db.fetch_events(tmp_path / "absent.db")
    assert df.empty and "camera_name" in df.columns and "composition" in df.columns
