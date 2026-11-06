"""Temporal event engine.

Turns noisy per-frame detections into confirmed alerts with a per-spot
state machine:  normal -> suspected -> confirmed -> cleared -> normal

Why: a detector can flicker (flag 1 frame in 10). Alerting on every flag
would spam operators. Requiring a high share of flagged frames across a
sliding window removes the flicker.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

NORMAL, SUSPECTED, CONFIRMED, CLEARED = "normal", "suspected", "confirmed", "cleared"


def severity_label(score: float, medium: float = 0.05, high: float = 0.15) -> str:
    if score >= high:
        return "high"
    if score >= medium:
        return "medium"
    return "low"


class SpotState:
    """State machine for ONE fixed camera spot.

    update(flagged, t, area) returns an event dict on a transition to
    "confirmed" or "cleared", otherwise None.
    """

    def __init__(self, window: int = 20, on_ratio: float = 0.7, off_ratio: float = 0.2,
                 suspect_ratio: float = 0.3, severity_medium: float = 0.05,
                 severity_high: float = 0.15):
        if window < 1:
            raise ValueError("window must be >= 1")
        if not 0 <= off_ratio < on_ratio <= 1:
            raise ValueError("need 0 <= off_ratio < on_ratio <= 1")
        self.window = window
        self.on_ratio, self.off_ratio, self.suspect_ratio = on_ratio, off_ratio, suspect_ratio
        self.sev_medium, self.sev_high = severity_medium, severity_high
        self.hist: deque = deque(maxlen=window)   # (t, flagged, area)
        self.state = NORMAL
        self._start: Optional[float] = None
        self._areas: list[float] = []
        self._last_flag_t: Optional[float] = None

    # ---- severity -------------------------------------------------------
    def _severity(self) -> dict:
        mean_area = sum(self._areas) / len(self._areas) if self._areas else 0.0
        duration = max(0.0, (self._last_flag_t or 0.0) - (self._start or 0.0))
        score = mean_area * (1 + duration / 60.0)   # bigger and longer = worse
        return {
            "severity": round(score, 4),
            "severity_label": severity_label(score, self.sev_medium, self.sev_high),
            "mean_area": round(mean_area, 4),
            "duration_s": round(duration, 2),
        }

    def current_event(self) -> Optional[dict]:
        """Info about the ongoing confirmed event (None if there isn't one)."""
        if self.state != CONFIRMED:
            return None
        return {"start_s": self._start, "end_s": self._last_flag_t, **self._severity()}

    # ---- main update ----------------------------------------------------
    def update(self, flagged: bool, t: float, area: float = 0.0) -> Optional[dict]:
        flagged = bool(flagged)
        area = float(area) if flagged else 0.0
        self.hist.append((t, flagged, area))

        if self.state == CONFIRMED and flagged:
            self._areas.append(area)
            self._last_flag_t = t

        if len(self.hist) < self.window:
            return None                      # not enough evidence yet

        ratio = sum(1 for _, f, _ in self.hist if f) / len(self.hist)

        if self.state == CLEARED:            # "cleared" lasts one step
            self.state = NORMAL

        if self.state in (NORMAL, SUSPECTED):
            if ratio >= self.on_ratio:
                flagged_entries = [(ts, a) for ts, f, a in self.hist if f]
                self._start = flagged_entries[0][0]          # earliest flagged frame
                self._areas = [a for _, a in flagged_entries]
                self._last_flag_t = flagged_entries[-1][0]
                self.state = CONFIRMED
                return {"type": CONFIRMED, "t": t, "start_s": self._start, **self._severity()}
            self.state = SUSPECTED if ratio >= self.suspect_ratio else NORMAL
            return None

        if self.state == CONFIRMED and ratio <= self.off_ratio:
            ev = {"type": CLEARED, "t": t, "start_s": self._start,
                  "end_s": self._last_flag_t, **self._severity()}
            self.state = CLEARED
            self._start, self._areas, self._last_flag_t = None, [], None
            return ev
        return None


def replay(trace, **engine_kwargs) -> list[dict]:
    """Re-run the engine over a saved trace [[t, flagged, area], ...].

    Lets you tune window / thresholds without re-running the detector.
    Returns events: {start_s, end_s (None if still open), severity, severity_label}.
    """
    st = SpotState(**engine_kwargs)
    events: list[dict] = []
    for t, flagged, area in trace:
        ev = st.update(flagged, t, area)
        if ev is None:
            continue
        if ev["type"] == CONFIRMED:
            events.append({"start_s": ev["start_s"], "end_s": None,
                           "severity": ev["severity"], "severity_label": ev["severity_label"]})
        else:
            events[-1].update(end_s=ev["end_s"], severity=ev["severity"],
                              severity_label=ev["severity_label"])
    cur = st.current_event()
    if cur and events and events[-1]["end_s"] is None:
        events[-1].update(severity=cur["severity"], severity_label=cur["severity_label"])
    return events


def raw_segments(trace) -> list[dict]:
    """Naive baseline: every run of consecutive flagged samples is an 'alert'
    (no smoothing). Used to show how many false alerts the engine removes."""
    segs: list[dict] = []
    cur = None
    for t, flagged, _ in trace:
        if flagged:
            if cur is None:
                cur = {"start_s": t, "end_s": t}
            else:
                cur["end_s"] = t
        elif cur is not None:
            segs.append(cur)
            cur = None
    if cur is not None:
        segs.append(cur)
    return segs
