"""Helper for tests only: builds a small SQLite database with fake events so the
dashboard and DB code can be tested. Not used by the application itself."""
import json
import random
import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np

from src import db
from src.config import load_config, rel_or_abs

MIXES = [{"plastic": 0.6, "metal": 0.25, "paper": 0.15}, {"glass": 0.7, "plastic": 0.3},
         {"organic": 1.0}, None]


def seed(db_path: Path, snap_dir: Path, n: int = 10, seed_value: int = 3) -> int:
    rng = random.Random(seed_value)
    cfg = load_config()
    snap_dir.mkdir(parents=True, exist_ok=True)
    conn = db.connect(db_path)
    for c in cfg["cameras"]:
        db.upsert_camera(conn, c)
    now = time.time()
    for i in range(n):
        cam = rng.choice(cfg["cameras"])
        start = now - rng.uniform(3600, 5 * 86400)
        sev = round(rng.uniform(0.02, 0.4), 3)
        label = "high" if sev >= 0.15 else "medium" if sev >= 0.05 else "low"
        img = np.full((90, 160, 3), 80, np.uint8)
        path = snap_dir / f"t{i}.jpg"
        cv2.imwrite(str(path), img)
        eid = db.open_event(conn, cam["id"], "garbage_dump", sev, label,
                            db.iso(start), rel_or_abs(path), MIXES[i % len(MIXES)])
        if i % 4:
            db.close_event(conn, eid, db.iso(start + 1800), sev, label)
    conn.close()
    return n


def raw_row(db_path: Path):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT composition FROM events ORDER BY id LIMIT 1").fetchone()[0]
    finally:
        conn.close()


def parsed(db_path: Path):
    return json.loads(raw_row(db_path))
