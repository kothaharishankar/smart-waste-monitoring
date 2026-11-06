import pytest

from src.events import SpotState, raw_segments, replay


def feed(st, flags, area=0.1, t0=0):
    out = []
    for i, f in enumerate(flags):
        ev = st.update(f, t0 + i, area if f else 0)
        if ev:
            out.append(ev)
    return out


def make(**kw):
    return SpotState(window=5, on_ratio=0.6, off_ratio=0.2, **kw)


def test_no_decision_before_window_is_full():
    st = make()
    assert feed(st, [True] * 4) == []
    assert st.state in ("normal", "suspected")


def test_confirms_after_sustained_detection():
    st = make()
    evs = feed(st, [True] * 5)
    assert [e["type"] for e in evs] == ["confirmed"]
    assert evs[0]["start_s"] == 0
    assert st.state == "confirmed"


def test_flicker_never_confirms():
    st = SpotState(window=10, on_ratio=0.7, off_ratio=0.2)
    assert feed(st, [True, False] * 20) == []
    assert st.state != "confirmed"


def test_clears_and_reports_last_flagged_time():
    st = make()
    evs = feed(st, [True] * 8 + [False] * 8)
    assert [e["type"] for e in evs] == ["confirmed", "cleared"]
    assert evs[1]["end_s"] == 7


def test_can_alert_again_after_clearing():
    st = make()
    evs = feed(st, [True] * 6 + [False] * 8 + [True] * 8)
    assert [e["type"] for e in evs] == ["confirmed", "cleared", "confirmed"]


def test_severity_scales_with_area():
    big, small = make(), make()
    assert feed(big, [True] * 5, area=0.4)[0]["severity_label"] == "high"
    assert feed(small, [True] * 5, area=0.01)[0]["severity_label"] == "low"


def test_severity_grows_with_duration():
    st = make()
    feed(st, [True] * 130, area=0.05)
    cur = st.current_event()
    assert cur["severity"] > 0.05 * 2          # 130 s -> factor > 3
    assert cur["end_s"] == 129


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        SpotState(window=5, on_ratio=0.3, off_ratio=0.5)
    with pytest.raises(ValueError):
        SpotState(window=0)


def test_replay_matches_live_engine():
    trace = [[i, 3 <= i <= 11, 0.1 if 3 <= i <= 11 else 0.0] for i in range(30)]
    events = replay(trace, window=5, on_ratio=0.6, off_ratio=0.2)
    assert len(events) == 1
    assert events[0]["start_s"] == 3 and events[0]["end_s"] == 11


def test_replay_keeps_open_event_open():
    trace = [[i, True, 0.1] for i in range(10)]
    events = replay(trace, window=5, on_ratio=0.6, off_ratio=0.2)
    assert len(events) == 1 and events[0]["end_s"] is None


def test_raw_segments_split_on_gaps():
    trace = [[0, False, 0], [1, True, .1], [2, False, 0], [3, True, .1], [4, True, .1]]
    assert raw_segments(trace) == [{"start_s": 1, "end_s": 1}, {"start_s": 3, "end_s": 4}]
