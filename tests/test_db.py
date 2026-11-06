import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_test_db import parsed, seed  # noqa: E402

from src import db  # noqa: E402
from src.segregate import average_composition, dominant, parse_composition  # noqa: E402


def test_old_database_is_migrated_and_keeps_its_rows(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE cameras (id TEXT PRIMARY KEY, name TEXT, lat REAL, lon REAL);"
        "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, camera_id TEXT, type TEXT, "
        "severity REAL, severity_label TEXT, start_ts TEXT, end_ts TEXT, snapshot_path TEXT);"
        "INSERT INTO events(camera_id, type, severity, severity_label, start_ts) "
        "VALUES('cam_01','garbage_dump',0.1,'medium','2026-10-01T10:00:00');")
    conn.commit()
    conn.close()

    c = db.connect(path)                       # migrates
    c.close()
    df = db.fetch_events(path)
    assert len(df) == 1 and df.loc[0, "type"] == "garbage_dump"
    assert "composition" in df.columns and df.loc[0, "composition"] is None


def test_composition_round_trip_and_update(tmp_path):
    path = tmp_path / "t.db"
    conn = db.connect(path)
    db.upsert_camera(conn, {"id": "cam_01", "name": "C1", "lat": 1.0, "lon": 2.0})
    eid = db.open_event(conn, "cam_01", "garbage_dump", 0.1, "medium",
                        "2026-10-01T10:00:00", "x.jpg", {"plastic": 0.7, "metal": 0.3})
    db.update_severity(conn, eid, 0.2, "high")                         # no composition: keep old
    db.close_event(conn, eid, "2026-10-01T11:00:00", 0.2, "high", {"plastic": 1.0})
    conn.close()

    df = db.fetch_events(path)
    comp = parse_composition(df.loc[0, "composition"])
    assert comp == {"plastic": 1.0} and dominant(comp) == "plastic"
    assert df.loc[0, "end_ts"] == "2026-10-01T11:00:00"


def test_update_without_composition_keeps_existing(tmp_path):
    path = tmp_path / "t.db"
    conn = db.connect(path)
    eid = db.open_event(conn, "cam_01", "garbage_dump", 0.1, "low", "2026-10-01T10:00:00", "x.jpg",
                        {"glass": 1.0})
    db.update_severity(conn, eid, 0.3, "high")
    conn.close()
    assert parse_composition(db.fetch_events(path).loc[0, "composition"]) == {"glass": 1.0}


def test_seed_helper_builds_a_usable_database(tmp_path):
    path = tmp_path / "s.db"
    assert seed(path, tmp_path / "snaps", n=8) == 8
    df = db.fetch_events(path)
    assert len(df) == 8 and df["end_ts"].isna().sum() >= 1
    assert parsed(path) == {"plastic": 0.6, "metal": 0.25, "paper": 0.15}
    avg = average_composition(df["composition"].tolist())
    assert avg and abs(sum(avg.values()) - 1.0) < 0.02
