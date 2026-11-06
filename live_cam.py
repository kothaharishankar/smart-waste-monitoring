import cv2
import numpy as np

NAMES = ["cardboard", "glass", "metal", "organic", "other", "paper", "plastic", "bin"]
CONF, IOU, S = 0.25, 0.45, 640
EVERY = 5  # run the model on every 5th frame (CPU is slow); boxes stay on screen in between

net = cv2.dnn.readNetFromONNX("models/best.onnx")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("Camera not found. Try index 1: VideoCapture(1, cv2.CAP_DSHOW)")

print("Camera open. First detection can take several seconds. Press q in the window to quit.")
n = 0
shapes = []  # (x, y, w, h, label)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    if n % EVERY == 0:
        h, w = frame.shape[:2]
        r = min(S / h, S / w)
        nw, nh = int(w * r), int(h * r)
        px, py = (S - nw) // 2, (S - nh) // 2
        canvas = np.full((S, S, 3), 114, np.uint8)
        canvas[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh))
        net.setInput(cv2.dnn.blobFromImage(canvas, 1 / 255.0, (S, S), swapRB=True))
        out = net.forward()[0].T
        scores = out[:, 4:]
        cls, conf = scores.argmax(1), scores.max(1)
        keep = conf > CONF
        out, cls, conf = out[keep], cls[keep], conf[keep]
        boxes = [[int((cx - bw / 2 - px) / r), int((cy - bh / 2 - py) / r), int(bw / r), int(bh / r)]
                 for cx, cy, bw, bh in out[:, :4]]
        shapes = []
        if boxes:
            for i in np.array(cv2.dnn.NMSBoxes(boxes, conf.tolist(), CONF, IOU)).flatten():
                x, y, bw, bh = boxes[i]
                shapes.append((x, y, bw, bh, f"{NAMES[cls[i]]} {conf[i]:.2f}"))
    n += 1
    for x, y, bw, bh, label in shapes:
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Waste detector (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()