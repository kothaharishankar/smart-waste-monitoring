"""Frame analysis: WHAT waste is visible (material by material) and WHERE it is
relative to bins, which decides the alert type.

One detector finds both bins and waste items. Per frame:

  * waste whose centre is INSIDE a bin box          -> normal, a bin is supposed to hold waste
  * waste just OUTSIDE a bin (within `bin_pad`)     -> overflow around that bin
        >= `min_overflow_items` such items          -> alert type "bin_overflowing"
  * waste far from any bin ("loose")                -> counts towards an illegal dump
        >= `min_items` items or >= `min_area` of frame -> alert type "garbage_dump"

Every alert also carries the material mix of the waste that caused it.
Used by the video pipeline, the dashboard and scripts/predict_image.py.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

import cv2

from .detector import Detection, draw  # noqa: F401  (Detection re-exported for convenience)

OVERFLOW = "bin_overflowing"
DUMP = "garbage_dump"
ACCUMULATION = DUMP            # older name kept so existing imports keep working


@dataclass
class FrameInfo:
    flagged: bool
    label: str                     # alert type if this frame is part of an alert
    area: float                    # 0-1, share of the frame covered by the problem waste
    count: int                     # number of waste items visible (all of them)
    comp: dict = field(default_factory=dict)          # material -> frame-area share, ALL visible waste
    counts: dict = field(default_factory=dict)        # material -> number of items, ALL visible waste
    kept: list = field(default_factory=list)          # detections that were actually used
    bins: list = field(default_factory=list)          # [{box, conf, overflowing, overflow_items}]
    problem_comp: dict = field(default_factory=dict)  # material -> frame-area share, waste CAUSING the alert


def _box_area(box) -> float:
    x1, y1, x2, y2 = box
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def _center(box):
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _inside(pt, box) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def _expand(box, pad: float):
    w, h = box[2] - box[0], box[3] - box[1]
    return (box[0] - pad * w, box[1] - pad * h, box[2] + pad * w, box[3] + pad * h)


def _in_roi(box, roi, w, h) -> bool:
    if not roi:
        return True
    cx, cy = _center(box)
    x1, y1, x2, y2 = roi
    return x1 <= cx / w <= x2 and y1 <= cy / h <= y2


def analyse_frame(dets, cfg: dict, frame_shape, roi=None) -> FrameInfo:
    """Decide whether one frame shows an overflowing bin and/or a garbage dump,
    and what the waste is made of."""
    ccfg, pcfg = cfg["classes"], cfg["pipeline"]
    waste_set = set(ccfg.get("waste_classes", []))
    bin_set = set(ccfg.get("bin_classes", []))
    direct_set = set(ccfg.get("direct_flag_classes", []))
    min_conf = float(pcfg["flag_conf"])
    min_items = max(1, int(pcfg.get("min_items", 3)))
    min_area = float(pcfg.get("min_area", 0.02))
    pad = float(pcfg.get("bin_pad", 0.35))
    min_overflow = max(1, int(pcfg.get("min_overflow_items", 2)))
    h, w = frame_shape[:2]
    frame_area = float(w * h)

    kept = [d for d in dets
            if d.conf >= min_conf and _in_roi(d.box, roi, w, h)
            and (d.label in waste_set or d.label in bin_set or d.label in direct_set)]
    waste = [d for d in kept if d.label in waste_set]
    bins = [d for d in kept if d.label in bin_set]
    direct = [d for d in kept if d.label in direct_set]

    # Where is each waste item relative to the bins?
    near = [[] for _ in bins]      # waste just outside bin i
    loose = []                     # waste that belongs to no bin
    for d in waste:
        c = _center(d.box)
        if any(_inside(c, b.box) for b in bins):
            continue               # inside a bin: that is what bins are for
        for i, b in enumerate(bins):
            if _inside(c, _expand(b.box, pad)):
                near[i].append(d)
                break
        else:
            loose.append(d)

    bin_info, overflow_items = [], []
    for b, items in zip(bins, near):
        over = len(items) >= min_overflow
        bin_info.append({"box": b.box, "conf": b.conf, "overflowing": over,
                         "overflow_items": len(items)})
        if over:
            overflow_items.extend(items)

    loose_area = sum(_box_area(d.box) for d in loose) / frame_area
    loose_flag = len(loose) >= min_items or loose_area >= min_area
    problem = overflow_items + (loose if loose_flag else [])

    comp: dict = {}
    counts: dict = {}
    for d in waste:
        comp[d.label] = comp.get(d.label, 0.0) + _box_area(d.box) / frame_area
        counts[d.label] = counts.get(d.label, 0) + 1
    problem_comp: dict = {}
    for d in problem:
        problem_comp[d.label] = problem_comp.get(d.label, 0.0) + _box_area(d.box) / frame_area

    flagged = bool(direct) or bool(overflow_items) or loose_flag
    if direct:
        label = max(direct, key=lambda d: _box_area(d.box)).label
    elif overflow_items:
        label = OVERFLOW
    else:
        label = DUMP
    direct_area = sum(_box_area(d.box) for d in direct) / frame_area
    area = min(1.0, max(sum(problem_comp.values()), direct_area)) if flagged else 0.0
    return FrameInfo(flagged=flagged, label=label, area=area, count=len(waste), comp=comp,
                     counts=counts, kept=kept, bins=bin_info,
                     problem_comp=problem_comp if flagged else {})


def shares(comp: dict) -> dict:
    """Normalise {material: amount} to shares that sum to ~1, largest first."""
    total = sum(v for v in comp.values() if v > 0)
    if total <= 0:
        return {}
    items = sorted(((k, v / total) for k, v in comp.items() if v > 0), key=lambda kv: -kv[1])
    return {k: round(v, 3) for k, v in items}


def dominant(comp: dict):
    """Most common material, or None."""
    return max(comp, key=comp.get) if comp else None


def parse_composition(value) -> dict:
    """Composition as stored in SQLite (JSON text) -> dict. Never raises."""
    if isinstance(value, dict):
        return value
    if not value or not isinstance(value, str):
        return {}
    try:
        d = json.loads(value)
    except (ValueError, TypeError):
        return {}
    return d if isinstance(d, dict) else {}


def average_composition(items) -> dict:
    """Mean material share over a list of compositions (events without one are ignored)."""
    comps = [c for c in (parse_composition(i) for i in items) if c]
    if not comps:
        return {}
    acc: Counter = Counter()
    for c in comps:
        acc.update(c)
    return shares({k: v / len(comps) for k, v in acc.items()})


def summary_text(info: FrameInfo) -> str:
    text = ("no waste detected" if not info.counts else
            ", ".join(f"{k} x{v}" for k, v in sorted(info.counts.items(), key=lambda kv: -kv[1])))
    if info.bins:
        n_over = sum(1 for b in info.bins if b["overflowing"])
        text += f" | bins {len(info.bins)} ({n_over} overflowing)"
    return text


def annotate_frame(frame, info: FrameInfo, cfg: dict, banner=None):
    """Annotated copy: colour-coded boxes per class, plus a red frame on overflowing bins."""
    img = draw(frame, info.kept, cfg["classes"].get("direct_flag_classes", []), banner)
    for b in info.bins:
        if b["overflowing"]:
            x1, y1, x2, y2 = b["box"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(img, f"OVERFLOWING ({b['overflow_items']})", (x1, min(img.shape[0] - 6, y2 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img


def analyse_image(detector, image, cfg: dict, roi=None):
    """Run the detector on ONE image. Returns (annotated_image, FrameInfo)."""
    info = analyse_frame(detector.detect(image), cfg, image.shape, roi)
    status = info.label.upper() if info.flagged else "ok"
    return annotate_frame(image, info, cfg, f"{status} | {summary_text(info)}"), info
