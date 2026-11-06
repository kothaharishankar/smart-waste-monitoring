import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from make_test_db import seed  # noqa: E402


def test_dashboard_renders_with_events(tmp_path, monkeypatch):
    db_path = tmp_path / "t.db"
    seed(db_path, tmp_path / "snaps", n=10)
    monkeypatch.setenv("WASTE_DB", str(db_path))
    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value.endswith("Smart Waste Monitoring")
    assert len(at.metric) == 4


def test_dashboard_renders_with_empty_database(tmp_path, monkeypatch):
    monkeypatch.setenv("WASTE_DB", str(tmp_path / "missing.db"))
    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("No alerts yet" in i.value for i in at.info)
