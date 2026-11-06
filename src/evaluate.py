"""Event-level evaluation: how many real incidents were caught, and how many
false / duplicate alerts did an operator get, with and without smoothing?

Ground truth CSV (data/ground_truth.csv), one row per incident:
    video,start_s,end_s
    clip1.mp4,35,140
    clip2.mp4,,            <- a clip with NO incident (leave start/end empty)

Usage:
    python -m src.evaluate --gt data/ground_truth.csv --videos-dir data/videos
    python -m src.evaluate --gt data/ground_truth.csv --sweep      # tune window / on_ratio
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .config import load_config, resolve
from .events import raw_segments, replay


def load_ground_truth(path) -> dict[str, list[tuple[float, float]]]:
    gt: dict[str, list[tuple[float, float]]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            video = row["video"].strip()
            gt.setdefault(video, [])
            s, e = (row.get("start_s") or "").strip(), (row.get("end_s") or "").strip()
            if s and e:
                gt[video].append((float(s), float(e)))
    return gt


def _overlaps(a_start, a_end, g_start, g_end, tol) -> bool:
    a_end = float("inf") if a_end is None else a_end
    return a_start <= g_end + tol and a_end >= g_start - tol


def score_alerts(alerts: list[dict], gt: list[tuple[float, float]], duration_s: float,
                 tol_s: float = 5.0) -> dict:
    """Match alerts to ground-truth incidents.

    - first alert overlapping an incident      -> true positive
    - extra alerts for an already-caught one   -> duplicate (counted as false alert)
    - alert overlapping nothing                -> false alert
    - incident with no alert                   -> missed
    """
    matched: set[int] = set()
    tp = unmatched = dup = 0
    for a in sorted(alerts, key=lambda x: x["start_s"]):
        hits = [i for i, (gs, ge) in enumerate(gt)
                if _overlaps(a["start_s"], a.get("end_s"), gs, ge, tol_s)]
        if not hits:
            unmatched += 1
        elif any(i not in matched for i in hits):
            matched.add(next(i for i in hits if i not in matched))
            tp += 1
        else:
            dup += 1
    return {"alerts": len(alerts), "tp": tp, "false_alerts": unmatched, "duplicates": dup,
            "incidents": len(gt), "detected": len(matched), "missed": len(gt) - len(matched),
            "duration_s": duration_s}


def summarize(rows: list[dict]) -> dict:
    s = {k: sum(r[k] for r in rows) for k in
         ("alerts", "tp", "false_alerts", "duplicates", "incidents", "detected", "missed", "duration_s")}
    bad = s["false_alerts"] + s["duplicates"]
    hours = s["duration_s"] / 3600.0
    s["precision"] = round(s["tp"] / s["alerts"], 3) if s["alerts"] else 0.0
    s["recall"] = round(s["detected"] / s["incidents"], 3) if s["incidents"] else 0.0
    s["bad_alerts_per_hour"] = round(bad / hours, 1) if hours else 0.0
    return s


def run_method(name, traces: dict, gt: dict, alert_fn, tol_s: float) -> list[dict]:
    rows = []
    for video, tr in traces.items():
        m = score_alerts(alert_fn(tr["trace"]), gt.get(video, []), tr["duration_s"], tol_s)
        rows.append({"method": name, "video": video, **m})
    return rows


def _load_or_make_traces(videos: list[str], args, cfg) -> dict:
    traces_dir = Path(args.traces_dir) if args.traces_dir else resolve(cfg["paths"]["traces_dir"])
    videos_dir = Path(args.videos_dir)
    detector, out = None, {}
    for v in videos:
        cache = traces_dir / f"{v}.json"
        if cache.exists() and not args.refresh:
            with open(cache, encoding="utf-8") as f:
                out[v] = json.load(f)
            continue
        from .detector import Detector
        from .pipeline import process_video, save_trace
        if detector is None:
            m = cfg["model"]
            detector = Detector(resolve(args.weights or m["weights"]), m["conf"], m["imgsz"], m.get("device"))
        print(f"Running detector on {v} ...")
        res = process_video(videos_dir / v, detector, cfg)
        save_trace(res, traces_dir, v)
        with open(cache, encoding="utf-8") as f:
            out[v] = json.load(f)
    return out


def _md_table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="ground-truth CSV")
    ap.add_argument("--videos-dir", default="data/videos")
    ap.add_argument("--traces-dir", default=None)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--tol", type=float, default=5.0, help="seconds of slack when matching alerts to incidents")
    ap.add_argument("--refresh", action="store_true", help="re-run the detector even if a cached trace exists")
    ap.add_argument("--sweep", action="store_true", help="grid-search window and on_ratio")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    gt = load_ground_truth(args.gt)
    traces = _load_or_make_traces(sorted(gt), args, cfg)
    ecfg = cfg["events"]

    rows = (run_method("raw (no smoothing)", traces, gt, raw_segments, args.tol)
            + run_method("event engine", traces, gt, lambda tr: replay(tr, **ecfg), args.tol))

    summary = {m: summarize([r for r in rows if r["method"] == m])
               for m in ("raw (no smoothing)", "event engine")}
    headers = ["method", "alerts", "true", "false", "duplicate", "incidents", "missed",
               "precision", "recall", "bad alerts/hour"]
    table = _md_table(headers, [
        [m, s["alerts"], s["tp"], s["false_alerts"], s["duplicates"], s["incidents"],
         s["missed"], s["precision"], s["recall"], s["bad_alerts_per_hour"]]
        for m, s in summary.items()])
    print("\n" + table)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "event_metrics_per_video.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["method"])
        w.writeheader()
        w.writerows(rows)
    (out_dir / "event_metrics.md").write_text(
        "# Event-level results\n\n" + table +
        f"\n\nSettings: window={ecfg['window']}, on_ratio={ecfg['on_ratio']}, "
        f"off_ratio={ecfg['off_ratio']}, tolerance={args.tol}s\n", encoding="utf-8")
    print(f"\nSaved to {out_dir}/event_metrics.md and event_metrics_per_video.csv")

    if args.sweep:
        sweep_rows = []
        for window in (5, 10, 20, 30):
            for on_ratio in (0.5, 0.7, 0.9):
                kw = {**ecfg, "window": window, "on_ratio": on_ratio}
                if kw["off_ratio"] >= on_ratio:
                    continue
                s = summarize(run_method("x", traces, gt, lambda tr: replay(tr, **kw), args.tol))
                sweep_rows.append([window, on_ratio, s["precision"], s["recall"], s["bad_alerts_per_hour"]])
        sweep_table = _md_table(["window", "on_ratio", "precision", "recall", "bad alerts/hour"], sweep_rows)
        print("\nSweep:\n" + sweep_table)
        (out_dir / "sweep.md").write_text("# Parameter sweep\n\n" + sweep_table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
