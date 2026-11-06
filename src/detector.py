"""YOLOv8 wrapper. ultralytics is imported lazily so the rest of the code
(event engine, tests) works even before it is installed."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class Detection:
    label: str
    conf: float
    box: tuple  # (x1, y1, x2, y2) in pixels


class Detector:
    def __init__(self, weights: str | Path, conf: float = 0.25, imgsz: int = 640, device=None):
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"Model weights not found at '{weights}'. Train first with "
                "`python scripts/train.py` and copy best.pt into models/ "
                "(or pass --weights)."
            )
        from ultralytics import YOLO  # lazy import

        self.model = YOLO(str(weights))
        self.names = self.model.names
        self.conf, self.imgsz, self.device = conf, imgsz, device

    def detect(self, frame) -> list[Detection]:
        res = self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False
        )[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            out.append(Detection(self.names[int(b.cls[0])], float(b.conf[0]), (x1, y1, x2, y2)))
        return out


# BGR colours so every material is easy to tell apart in an annotated frame.
MATERIAL_COLORS = {
    "plastic": (0, 140, 255),      # orange
    "metal": (200, 200, 200),      # light grey
    "glass": (255, 200, 0),        # light blue
    "paper": (0, 220, 255),        # yellow
    "cardboard": (60, 120, 180),   # brown
    "organic": (0, 180, 0),        # green
    "other": (160, 0, 160),        # purple
    "bin": (255, 255, 255),        # white
    "bin_ok": (0, 180, 0),
    "bin_overflowing": (0, 0, 255),
    "garbage_dump": (0, 0, 255),
}


def color_for(label: str) -> tuple:
    """Fixed colour for known classes, a stable fallback for unknown ones."""
    if label in MATERIAL_COLORS:
        return MATERIAL_COLORS[label]
    h = sum(ord(c) for c in label)
    return (50 + (h * 37) % 180, 50 + (h * 91) % 180, 50 + (h * 53) % 180)


def draw(frame, detections, flag_classes=(), banner: str | None = None):
    """Return an annotated copy of the frame. Each material has its own colour;
    classes in `flag_classes` get a thicker box."""
    img = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = d.box
        color = color_for(d.label)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if d.label in flag_classes else 2)
        cv2.putText(img, f"{d.label} {d.conf:.2f}", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if banner:
        cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(img, banner, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img
