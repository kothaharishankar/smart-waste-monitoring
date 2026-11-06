# Smart Waste Monitoring

Detects **bins** and **waste by material** (cardboard, glass, metal, organic, other, paper, plastic) in photos, video, webcam or IP-camera streams with one YOLOv8 model. Simple geometric rules turn the detections into **bin-overflow** and **garbage-dump** alerts, which are logged to SQLite and shown in a Streamlit dashboard.

```
photo / video / webcam / RTSP
   -> OpenCV reads frames (1 per second)
   -> YOLOv8 detector: boxes for bins and for each waste item (with its material)
   -> frame rules: where is each item relative to the bins?
   -> event engine: sliding window + state machine (normal -> suspected -> confirmed -> cleared)
   -> SQLite (alert, severity, waste mix, snapshot)  ->  Streamlit dashboard
```

## How alerts are decided

| Where the waste is | Meaning | Alert |
| --- | --- | --- |
| Inside a bin box | Normal, bins hold waste | none |
| Just outside a bin (at least `min_overflow_items` items within `bin_pad` of the bin box) | Bin is overflowing | `bin_overflowing` |
| Away from any bin (at least `min_items` items, or at least `min_area` of the frame) | Illegal dump | `garbage_dump` |

A video detector flickers, so the event engine looks at the last 20 samples: an alert is **confirmed** when 70% are flagged, and **cleared** when 20% or fewer are. Severity = mean waste area x (1 + duration in seconds / 60).

## Results

Trained with YOLOv8s on Google Colab (T4 GPU): 30 epochs, batch 16, image size 640, AdamW, about 0.7 hours. Data: three merged public datasets (train 5873 / val 465 / test 284 images).

| Metric | Value |
| --- | --- |
| mAP50 | 0.822 |
| mAP50-95 | 0.597 |
| Precision | 0.876 |
| Recall | 0.741 |

| Class | mAP50 |
| --- | --- |
| bin | 0.995 |
| plastic | 0.885 |
| paper | 0.883 |
| metal | 0.856 |
| glass | 0.784 |
| organic | 0.740 |
| cardboard | 0.608 |

Progress: 7 materials 0.777, adding the bin class 0.785, adding an overflow dataset 0.822. See `results/` for the confusion matrix and experiment notes.

## Limitations (please read)

- **Cluttered street scenes are weak.** Piles of black bags, dumpsters and mixed rubbish were barely present in the training data, so the model often finds the bin but misses the waste around it. A larger model (YOLOv8m) did not help, so the limit is training data, not model size.
- **The bin score (0.995) is inflated.** Near-duplicate frames of the same scenes appear across splits (the dataset checker warns about this).
- Overflow is a geometric rule, not a learned class. Thresholds (`bin_pad`, `min_overflow_items`, `min_items`, `min_area`, `roi`) must be tuned per camera. It does not measure how full a closed bin is.
- If bins are missed in a frame, nearby waste is treated as loose and can raise a `garbage_dump` alert.
- Waste mix is a share of visible area, not weight or volume.
- Plastic, glass and metal are easily confused at a distance; cardboard is the weakest class (few examples).
- The severity score is not calibrated. No authentication, no REST API, local use only.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest
```

Tests use a fake detector, so no model is needed. If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. If `pip` fails on PyTorch, use Python 3.12 (`py -3.12 -m venv .venv`).

The trained weights are not stored in this repository. Train your own (below) and place the result at `models/best.pt`.

## Run

```powershell
python scripts\predict_image.py --source "C:\path\to\photos"     # annotated copies go to results\predictions
python -m src.pipeline --video "C:\path\to\clip.mp4" --out out.mp4
python -m src.pipeline --video 0 --show                           # webcam (q to quit)
python -m streamlit run app\dashboard.py                          # dashboard
```

Pipeline options: `--camera cam_02`, `--no-db`, `--max-seconds N`, RTSP via `--video rtsp://...`.

**Webcam without PyTorch (ONNX + OpenCV).** If PyTorch cannot run on your machine, export once with `YOLO("models/best.pt").export(format="onnx", imgsz=640, opset=12)`, put `best.onnx` in `models/`, and run `python live_cam.py`. It draws boxes only; the overflow and dump rules need the main pipeline.

## Build the dataset and train

1. Download datasets in YOLO (or COCO) format, one folder each.
2. Check how class names will be mapped (nothing is written):
   `python scripts\prepare_dataset.py --src "C:\path\ds1" --src "C:\path\ds2" --dry-run`
   Fix `configs\class_map.yaml` if any mapping looks wrong.
3. Build the merged dataset: same command with `--clean` instead of `--dry-run`.
4. Check it: `python scripts\check_dataset.py --root data` (instances per class, leakage warnings).
5. Train on Colab with `notebooks\train_colab.ipynb` (upload `data.zip` to Google Drive first), or locally with `python scripts\train.py --epochs 50`.
6. Copy `best.pt` to `models\best.pt`.

## Settings

All tunable values live in `configs/params.yaml`: detector confidence, `sample_fps`, flag confidence, overflow and dump thresholds, event window and ratios, severity thresholds, and cameras (`lat`, `lon`, optional `roi`).

## Project layout

```
app/dashboard.py          Streamlit UI (Alerts, Map, Trends, Analyse images, Analyse a video)
src/detector.py           YOLO wrapper + colour-coded drawing
src/segregate.py          per-frame rules: overflow / dump decision, material mix
src/events.py             sliding-window state machine
src/pipeline.py           video -> detect -> events -> DB / snapshots / trace
src/db.py                 SQLite helpers (cameras, events)
src/evaluate.py           event-level metrics
src/config.py             loads configs/params.yaml
scripts/                  prepare_dataset.py, check_dataset.py, train.py, predict_image.py
configs/                  params.yaml, class_map.yaml, data.yaml
notebooks/                train_colab.ipynb, autolabel_colab.ipynb (experimental)
live_cam.py               webcam demo with ONNX + OpenCV (no PyTorch)
tests/                    pytest suite
```

## Credits

- Detector: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
- Training data: three public datasets from [Roboflow Universe](https://universe.roboflow.com), used under CC BY 4.0: *waste_yolov*, *Trash_bin_detection* and *garbage can overflow*. Dataset images are not redistributed in this repository; please credit the original authors if you reuse them.
- Libraries: OpenCV, NumPy, pandas, PyYAML, Streamlit, folium, pytest.

## License

MIT, see `LICENSE`. Note that the Ultralytics dependency is AGPL-3.0.