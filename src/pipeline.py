"""video -> detector -> material analysis -> event engine -> SQLite + snapshots.

Usage (from the project root):
    python -m src.pipeline --video data/videos/clip1.mp4 --camera cam_01
    python -m src.pipeline --video 0                      # webcam
    python -m src.pipeline --video rtsp://...             # IP camera
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2

from . import db
from .config import get_camera, load_config, rel_or_abs, resolve
from .events import SpotState
from .segregate import DUMP, analyse_frame, annotate_frame, shares, summary_text


@dataclass
class RunResult:
    source: str
    camera_id: str
    sample_fps: float = 1.0
    frames_sampled: int = 0
    duration_s: float = 0.0
    trace: list = field(default_factory=list)    # [[t, flagged, area], ...]
    events: list = field(default_factory=list)   # dicts, times in video seconds


def _is_live(source) -> bool:
    s = str(source)
    return s.isdigit() or s.startswith("rtsp://")


def _save_snapshot(img, alerts_dir: Path, camera_id: str, epoch: float, label: str) -> str:
    alerts_dir.mkdir(parents=True, exist_ok=True)
    path = alerts_dir / f"{camera_id}_{int(epoch)}_{label}.jpg"
    cv2.imwrite(str(path), img)
    return rel_or_abs(path)


def process_video(source, detector, cfg: dict, camera: dict | None = None, conn=None,
                  alerts_dir: Path | None = None, out_video: str | Path | None = None,
                  out_fps: float = 5.0, base_epoch: float | None = None,
                  progress: Optional[Callable[[float], None]] = None,
                  show: bool = False, max_seconds: float | None = None) -> RunResult:
    """Run the full pipeline on a video file / webcam / RTSP stream.

    `detector` is anything with .detect(frame) -> list[Detection] (so tests can
    pass a fake). If `conn` is given, events are written to SQLite.
    """
    live = _is_live(source)
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    pcfg, ecfg = cfg["pipeline"], cfg["events"]
    sample_fps = float(pcfg["sample_fps"])
    interval = 1.0 / sample_fps
    cam = camera or get_camera(cfg)
    roi = cam.get("roi")
    alerts_dir = alerts_dir or resolve(cfg["paths"]["alerts_dir"])

    spot = SpotState(**ecfg)
    recent: deque = deque(maxlen=spot.window)    # (flagged, label, material areas) per sample
    comp_acc: Counter = Counter()                # material areas summed over the open event
    result = RunResult(source=str(source), camera_id=cam["id"], sample_fps=sample_fps)

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not live and (fps <= 0 or fps != fps):
        fps = 30.0
    step = 1 if live else max(1, int(round(fps / sample_fps)))
    total = 0 if live else (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    base = base_epoch if base_epoch is not None else time.time()

    writer = None
    idx, last_sample_t, t0 = 0, -1e9, time.time()
    try:
        while True:
            if live:
                ok, frame = cap.read()
                if not ok:
                    break
                t = time.time() - t0
                if t - last_sample_t < interval:
                    continue
                last_sample_t = t
            else:
                if not cap.grab():
                    break
                cur, idx = idx, idx + 1
                if cur % step:
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    break
                t = cur / fps
                if progress and total:
                    progress(min(1.0, cur / total))
            if max_seconds is not None and t > max_seconds:
                break

            info = analyse_frame(detector.detect(frame), cfg, frame.shape, roi)
            recent.append((info.flagged, info.label, info.problem_comp))
            result.trace.append([round(t, 3), info.flagged, round(info.area, 5)])
            result.frames_sampled += 1

            ev = spot.update(info.flagged, t, info.area)
            annotated = None
            if ev or out_video or show:
                banner = f"{cam['id']} | {spot.state} | {summary_text(info)}"
                annotated = annotate_frame(frame, info, cfg, banner)

            if ev and ev["type"] == "confirmed":
                flagged_recent = [(lab, c) for f, lab, c in recent if f]
                counts = Counter(lab for lab, _ in flagged_recent)
                label = counts.most_common(1)[0][0] if counts else DUMP
                comp_acc = Counter()
                for _, c in flagged_recent:
                    comp_acc.update(c)
                comp = shares(comp_acc)
                snap = _save_snapshot(annotated, alerts_dir, cam["id"], base + t, label)
                rec = {"type": label, "start_s": ev["start_s"], "end_s": None,
                       "severity": ev["severity"], "severity_label": ev["severity_label"],
                       "composition": comp, "snapshot": snap, "event_id": None}
                if conn is not None:
                    rec["event_id"] = db.open_event(
                        conn, cam["id"], label, ev["severity"], ev["severity_label"],
                        db.iso(base + ev["start_s"]), snap, comp)
                result.events.append(rec)
            elif ev and ev["type"] == "cleared":
                rec = result.events[-1]
                comp = shares(comp_acc)
                rec.update(end_s=ev["end_s"], severity=ev["severity"],
                           severity_label=ev["severity_label"], composition=comp)
                if conn is not None and rec["event_id"]:
                    db.close_event(conn, rec["event_id"], db.iso(base + ev["end_s"]),
                                   ev["severity"], ev["severity_label"], comp)
                comp_acc = Counter()
            elif spot.state == "confirmed" and info.flagged:
                comp_acc.update(info.problem_comp)

            if out_video and annotated is not None:
                if writer is None:
                    h, w = annotated.shape[:2]
                    Path(out_video).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                             out_fps, (w, h))
                writer.write(annotated)
            if show and annotated is not None:
                cv2.imshow("smart-waste-monitoring (q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    # event still open when the stream ended: refresh its severity and composition
    cur_ev = spot.current_event()
    if cur_ev and result.events and result.events[-1]["end_s"] is None:
        rec = result.events[-1]
        comp = shares(comp_acc)
        rec.update(severity=cur_ev["severity"], severity_label=cur_ev["severity_label"],
                   composition=comp)
        if conn is not None and rec["event_id"]:
            db.update_severity(conn, rec["event_id"], cur_ev["severity"],
                               cur_ev["severity_label"], comp)

    if result.trace:
        result.duration_s = float(result.trace[-1][0]) + interval
    if progress:
        progress(1.0)
    return result


def save_trace(result: RunResult, traces_dir: Path, name: str) -> Path:
    traces_dir.mkdir(parents=True, exist_ok=True)
    path = traces_dir / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"video": name, "duration_s": result.duration_s,
                   "sample_fps": result.sample_fps, "trace": result.trace}, f)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="video file, webcam index (0) or rtsp:// URL")
    ap.add_argument("--camera", default=None, help="camera id from params.yaml (default: first)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--weights", default=None, help="override model.weights")
    ap.add_argument("--out", default=None, help="write an annotated video here (e.g. results/demo.mp4)")
    ap.add_argument("--no-db", action="store_true", help="do not write events to SQLite")
    ap.add_argument("--show", action="store_true", help="live preview window")
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args(argv)

    from .detector import Detector

    cfg = load_config(args.config)
    cam = get_camera(cfg, args.camera)
    mcfg = cfg["model"]
    try:
        detector = Detector(resolve(args.weights or mcfg["weights"]), mcfg["conf"],
                            mcfg["imgsz"], mcfg.get("device"))
    except FileNotFoundError as e:
        raise SystemExit(f"ERROR: {e}")

    conn = None
    if not args.no_db:
        conn = db.connect(resolve(cfg["paths"]["db"]))
        for c in cfg["cameras"]:
            db.upsert_camera(conn, c)

    try:
        result = process_video(args.video, detector, cfg, camera=cam, conn=conn,
                               out_video=args.out, show=args.show, max_seconds=args.max_seconds)
    finally:
        if conn is not None:
            conn.close()

    if not _is_live(args.video):
        save_trace(result, resolve(cfg["paths"]["traces_dir"]), Path(args.video).name)

    print(f"Sampled {result.frames_sampled} frames ({result.duration_s:.0f}s of video)")
    print(f"Confirmed events: {len(result.events)}")
    for e in result.events:
        end = f"{e['end_s']:.0f}s" if e["end_s"] is not None else "ongoing"
        mix = ", ".join(f"{k} {v:.0%}" for k, v in e["composition"].items()) or "n/a"
        print(f"  - {e['type']:<20} {e['start_s']:.0f}s -> {end}  "
              f"severity={e['severity_label']} ({e['severity']})")
        print(f"      waste mix: {mix}")
        print(f"      snapshot : {e['snapshot']}")


if __name__ == "__main__":
    main()
