import json

from src import evaluate
from src.evaluate import score_alerts, summarize


def test_exact_match_is_true_positive():
    m = score_alerts([{"start_s": 12, "end_s": 28}], [(10, 30)], 100)
    assert (m["tp"], m["false_alerts"], m["duplicates"], m["missed"]) == (1, 0, 0, 0)


def test_duplicates_and_false_alerts_counted():
    alerts = [{"start_s": 12, "end_s": 13}, {"start_s": 15, "end_s": 16}, {"start_s": 90, "end_s": 91}]
    m = score_alerts(alerts, [(10, 30)], 100, tol_s=0)
    assert (m["tp"], m["duplicates"], m["false_alerts"], m["detected"]) == (1, 1, 1, 1)


def test_missed_incident():
    m = score_alerts([], [(10, 30)], 100)
    assert m["missed"] == 1 and m["detected"] == 0


def test_open_ended_alert_matches():
    m = score_alerts([{"start_s": 20, "end_s": None}], [(10, 30)], 100)
    assert m["tp"] == 1


def test_tolerance_allows_small_offset():
    a = [{"start_s": 33, "end_s": 40}]
    assert score_alerts(a, [(10, 30)], 100, tol_s=0)["tp"] == 0
    assert score_alerts(a, [(10, 30)], 100, tol_s=5)["tp"] == 1


def test_summary_rates():
    rows = [{"alerts": 4, "tp": 1, "false_alerts": 1, "duplicates": 2, "incidents": 1,
             "detected": 1, "missed": 0, "duration_s": 3600}]
    s = summarize(rows)
    assert s["precision"] == 0.25 and s["recall"] == 1.0 and s["bad_alerts_per_hour"] == 3.0


def test_cli_with_cached_traces(tmp_path, capsys):
    traces = tmp_path / "traces"
    traces.mkdir()
    # incident 10-40 s, plus a flicker at 50 s
    trace = [[t, 10 <= t <= 40 or t in (50, 52), 0.1] for t in range(60)]
    (traces / "a.mp4.json").write_text(json.dumps({"video": "a.mp4", "duration_s": 60, "trace": trace}))
    gt = tmp_path / "gt.csv"
    gt.write_text("video,start_s,end_s\na.mp4,10,40\n")
    evaluate.main(["--gt", str(gt), "--traces-dir", str(traces), "--out-dir", str(tmp_path / "out"), "--sweep"])
    out = capsys.readouterr().out
    assert "raw (no smoothing)" in out and "event engine" in out
    assert (tmp_path / "out" / "event_metrics.md").exists()
    assert (tmp_path / "out" / "sweep.md").exists()
