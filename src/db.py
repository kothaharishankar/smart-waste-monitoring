"""SQLite helpers: cameras + events (with material composition)."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
    id   TEXT PRIMARY KEY,
    name TEXT,
    lat  REAL,
    lon  REAL
);
CREATE TABLE IF NOT EXISTS events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id      TEXT,
    type           TEXT,      -- garbage_dump | bin_overflowing | garbage_dump
    severity       REAL,
    severity_label TEXT,      -- low | medium | high
    start_ts       TEXT,      -- ISO timestamp
    end_ts         TEXT,      -- NULL while the event is ongoing
    snapshot_path  TEXT,
    composition    TEXT,      -- JSON {"plastic": 0.55, "metal": 0.2, ...} share of waste area
    FOREIGN KEY (camera_id) REFERENCES cameras(id)
);
"""

EVENT_COLUMNS = ["id", "camera_id", "type", "severity", "severity_label", "start_ts",
                 "end_ts", "snapshot_path", "composition", "camera_name", "lat", "lon"]


def iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).isoformat(timespec="seconds")


def _migrate(conn) -> None:
    """Databases created by an older version lack the composition column."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(events)")}
    if cols and "composition" not in cols:
        conn.execute("ALTER TABLE events ADD COLUMN composition TEXT")
        conn.commit()


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _comp_json(composition) -> str | None:
    return json.dumps(composition) if composition else None


def upsert_camera(conn, cam: dict) -> None:
    conn.execute(
        "INSERT INTO cameras(id, name, lat, lon) VALUES(?,?,?,?) "
        "ON CONFLICT(id) DO UPDATE SET name=excluded.name, lat=excluded.lat, lon=excluded.lon",
        (cam["id"], cam.get("name"), cam.get("lat"), cam.get("lon")),
    )
    conn.commit()


def open_event(conn, camera_id, type_, severity, label, start_ts, snapshot_path,
               composition=None) -> int:
    cur = conn.execute(
        "INSERT INTO events(camera_id, type, severity, severity_label, start_ts, end_ts, "
        "snapshot_path, composition) VALUES(?,?,?,?,?,NULL,?,?)",
        (camera_id, type_, severity, label, start_ts, snapshot_path, _comp_json(composition)),
    )
    conn.commit()
    return cur.lastrowid


def update_severity(conn, event_id, severity, label, composition=None) -> None:
    conn.execute(
        "UPDATE events SET severity=?, severity_label=?, composition=COALESCE(?, composition) WHERE id=?",
        (severity, label, _comp_json(composition), event_id))
    conn.commit()


def close_event(conn, event_id, end_ts, severity, label, composition=None) -> None:
    conn.execute(
        "UPDATE events SET end_ts=?, severity=?, severity_label=?, "
        "composition=COALESCE(?, composition) WHERE id=?",
        (end_ts, severity, label, _comp_json(composition), event_id))
    conn.commit()


def fetch_events(db_path: str | Path) -> pd.DataFrame:
    """All events joined with camera info, newest first. Empty frame if no DB yet."""
    db_path = Path(db_path)
    if not db_path.exists():
        return pd.DataFrame(columns=EVENT_COLUMNS)
    conn = sqlite3.connect(db_path)
    try:
        _migrate(conn)
        return pd.read_sql_query(
            "SELECT e.id, e.camera_id, e.type, e.severity, e.severity_label, e.start_ts, "
            "e.end_ts, e.snapshot_path, e.composition, c.name AS camera_name, c.lat, c.lon "
            "FROM events e LEFT JOIN cameras c ON c.id = e.camera_id "
            "ORDER BY e.start_ts DESC",
            conn,
        )
    finally:
        conn.close()


def fetch_cameras(db_path: str | Path) -> pd.DataFrame:
    db_path = Path(db_path)
    if not db_path.exists():
        return pd.DataFrame(columns=["id", "name", "lat", "lon"])
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query("SELECT id, name, lat, lon FROM cameras", conn)
    finally:
        conn.close()
