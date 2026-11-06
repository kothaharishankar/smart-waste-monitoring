# Smart Waste Monitoring

Watches camera feeds and does two jobs with **one model**:

1. finds **overflowing bins** and **illegal garbage dumps**, and
2. **separates the waste by material** (plastic, metal, glass, paper, cardboard, organic, other).

Every alert says *what is wrong* and *what the waste is made of*. Alerts are logged to SQLite
and shown on a Streamlit dashboard with a map.

```
photo / video / webcam / RTSP
   -> YOLOv8 detector: boxes for BINS and for every waste item (labelled with its MATERIAL)
   -> frame analysis: where is each item relative to the bins? what is it made of?
   -> event engine: sliding window + state machine (normal -> suspected -> confirmed -> cleared)
   -> SQLite (alert + waste mix + snapshot)  ->  Streamlit dashboard
```

**How the alert type is decided** (per camera frame):

| Where the waste is | Meaning | Alert |
|---|---|---|
| inside a bin box | normal - bins are meant to hold waste | none |
| just outside a bin (>= `min_overflow_items` items within `bin_pad` of the bin) | the bin is overflowing | `bin_overflowing` |
| away from any bin (>= `min_items` items or >= `min_area` of the frame) | illegal dump | `garbage_dump` |

The alert's waste mix counts only the waste that caused it (the items outside the bin, or
the loose pile), so "bin overflowing - 70% plastic, 30% paper" is accurate.

Two ideas make it more than "a model":

1. **Material segregation** - every detected item carries a material class, so each alert says
   *what* is piling up ("55% plastic, 30% metal, 15% paper"), not just "garbage".
2. **The event engine** - a detector flickers. Requiring ~70% of a sliding window to be
   flagged turns noisy detections into a few real alerts. `src/evaluate.py` measures this:
   false/duplicate alerts per hour **with vs. without** the engine.

There is **no demo data anywhere**. Everything you see comes from your own dataset, model,
photos and videos.

---

## Part A - One-time setup

Open the folder that contains `requirements.txt` in VS Code, then in the terminal (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
```

`pytest` should pass (it uses a fake detector, so no model is needed). If activation is blocked:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. If `pip` fails on `torch`, use Python 3.12
(`py -3.12 -m venv .venv`) - very new Python versions often lack PyTorch wheels.

All commands below are run from this folder.

---

## Part B - Build your dataset (this is where the quality comes from)

### Step 1 - Download datasets in YOLO format

I checked these pages; verify the licence yourself before publishing anything.

| Dataset | Classes (as listed) | Size / format | Licence | Notes |
|---|---|---|---|---|
| [Roboflow `waste_yolov`](https://universe.roboflow.com/cera/waste_yolov) | paper, plastic, glass, metal, cardboard, waste, bio-degradable | ~1.2k images (more with augmentation) | CC BY 4.0 | Free Roboflow account; **Download -> YOLOv8**. Best starting point. |
| [Mendeley - Trash Detection Dataset](https://data.mendeley.com/datasets/z732f9pwxt) | BIODEGRADABLE, CARDBOARD, GLASS, METAL, PAPER, PLASTIC, ALL | boxes, YOLO format | not stated on the page | `ALL` is a catch-all class; the converter drops it. |
| [Mendeley - re-annotated TrashNet](https://data.mendeley.com/datasets/f6y37w8hxb) | Glass, Paper, Cardboard, Plastic, Metal, Trash | 2,527 images, polygons (converted to boxes) | not stated on the page | **Clean single-object photos** - useful as a base, but they look nothing like a street camera. |

Also search Roboflow Universe for "waste detection" / "garbage detection" and pick sets with
material classes. Unzip each download into its own folder.

**Do not rely on public sets alone.** Models trained on tidy photos drop badly on real
camera footage. Add **100-300 photos taken from your real camera angle** (different light,
weather, distance), label them in Roboflow (free) or CVAT with the **same class names**,
export as YOLOv8, and include them as another `--src`. This is the single biggest quality lever.

**Bins need labels too.** None of the three datasets above has a bin class, and the overflow
rule needs one. In your own photos, label **both**:

- every **bin** with class `bin` - draw the box around the *container itself* (rim to base), **not**
  around waste piled on top of or around it (that waste gets its own material boxes), and
- every **waste item** with its material (`plastic`, `metal`, ...).

Include photos of normal bins, overflowing bins and loose dumps. Label *everything* in each photo:
mixing in a bins-only dataset teaches the model that unlabelled waste is background. You may also
search Roboflow Universe for a bin / dustbin detection set (I have not verified any), but treat it
as a supplement. `scripts/prepare_dataset.py` already maps names like "dustbin" or "garbage bin" to `bin`.

### Step 2 - Merge everything into unified material classes

Look at how classes will be mapped first (nothing is written):

```powershell
python scripts/prepare_dataset.py --src "C:\path\to\dataset1" --src "C:\path\to\dataset2" --dry-run
```

Read the table. If a class maps wrongly, edit `configs/class_map.yaml` (add it under
`explicit:` or `drop:`) and repeat. Then build `data/`:

```powershell
python scripts/prepare_dataset.py --src "C:\path\to\dataset1" --src "C:\path\to\dataset2" --clean
```

This copies images/labels into `data/images|labels/{train,val,test}`, converts polygons to
boxes, drops unusable images, and rewrites `configs/data.yaml`.

### Step 3 - Check it

```powershell
python scripts/check_dataset.py --root data
```

Look at the instances per material. A material with a handful of examples will be detected
badly - collect more of it or merge it into `other`. Fix any "possible leakage" warning
(near-duplicate frames of the same scene in both train and val inflate your scores).

---

## Part C - Train

**Recommended: Google Colab (free GPU).**

1. Zip the data: `Compress-Archive -Path data\images, data\labels -DestinationPath data.zip`
2. Upload `data.zip` to the top level of Google Drive.
3. Open `notebooks/train_colab.ipynb` in Colab -> Runtime -> **T4 GPU** -> run the cells top to bottom.
4. Download `best.pt` into this project's `models/` folder (so it is `models/best.pt`).
5. Keep the confusion matrix image it downloads for your write-up.

Local alternative (slow without a GPU): `python scripts/train.py --epochs 50`
(copies `best.pt` to `models/`, appends to `results/experiments.md`, writes a per-material table).

Expect confusion between look-alike materials (plastic vs glass vs metal). That is normal -
report per-material precision/recall and the confusion matrix instead of hiding it.

---

## Part D - See the segregation on photos

```powershell
python scripts/predict_image.py --source "C:\path\to\photos"
```

Annotated copies go to `results/predictions/` (each material has its own colour), and the
console prints items and area share per material. You can do the same in the dashboard
(**Analyse images** tab).

Use this step to calibrate `configs/params.yaml` on photos from your camera:

| Setting | Meaning | How to set it |
|---|---|---|
| `pipeline.flag_conf` | ignore detections below this confidence | raise it if you see phantom items |
| `pipeline.min_items` / `min_area` | how much **loose** waste (not near a bin) counts as a dump | pick values that separate a clean spot from a dirty one |
| `pipeline.bin_pad` | size of the zone around a bin where waste counts as "overflow" (fraction of the bin's width/height) | raise it if overflow pile-ups spread wide |
| `pipeline.min_overflow_items` | waste items in that zone needed to call the bin overflowing | raise it if single stray items cause alerts |
| `cameras[].roi` | `[x1,y1,x2,y2]` (0-1): only count waste inside this box | uncomment and set it around the bin / dump area |
| `cameras[].lat/lon` | map position | enter the real coordinates |

---

## Part E - Run on video

Record 3-5 clips (1-3 minutes) from the camera angle you care about, including **at least one
clip with no pile-up**. Put them in `data/videos/`.

```powershell
python -m src.pipeline --video data/videos/clip1.mp4 --out results/clip1.mp4
```

It writes confirmed alerts (with waste mix) to `data/waste.db`, snapshots to `results/alerts/`,
an annotated video, and a trace file for evaluation. Webcam: `--video 0`. IP camera:
`--video rtsp://...`. Use `--camera cam_02` to choose another camera from `params.yaml`.

`events.window` counts samples: at `sample_fps: 1.0`, `window: 20` is 20 seconds, fine for
short test clips. For real deployments use minutes.

---

## Part F - Dashboard

```powershell
streamlit run app/dashboard.py
```

Tabs: **Alerts** (table, snapshot, waste mix) - **Map** - **Trends** (composition across
alerts, alerts per day / hour) - **Analyse images** - **Analyse a video**. Press *Refresh* in
the sidebar after running the pipeline.

---

## Part G - Measure it

Create `data/ground_truth.csv` listing the incidents you see in your clips (see
`data/ground_truth.example.csv`; an empty start/end row means "no incident in this clip"):

```powershell
python -m src.evaluate --gt data/ground_truth.csv --sweep
```

`results/event_metrics.md` compares **raw detections vs. the event engine**: alerts, false and
duplicate alerts, missed incidents, precision, recall, bad alerts per hour. `--sweep` tries
different window sizes and thresholds on the saved traces without re-running the model.

---

## Part H - Publish

Fill in your real numbers in this file, add a screen recording and two screenshots (an
annotated photo and the dashboard), and push to GitHub. `.gitignore` already excludes data,
weights and databases. Blur faces and number plates in any photo you publish.

---

## Project layout

```
configs/params.yaml        thresholds, window sizes, cameras (edit lat/lon!)
configs/data.yaml          class names + split paths (written by prepare_dataset.py)
configs/class_map.yaml     how downloaded class names map to your materials
src/detector.py            YOLO wrapper + colour-coded drawing
src/segregate.py           per-frame analysis: bins vs waste, overflow/dump decision, material mix
src/events.py              sliding-window state machine, replay(), raw baseline
src/pipeline.py            video -> detect -> events -> DB / snapshots / trace
src/db.py                  SQLite helpers
src/evaluate.py            event-level metrics + parameter sweep
app/dashboard.py           Streamlit UI
scripts/                   prepare_dataset.py, check_dataset.py, train.py, predict_image.py
notebooks/train_colab.ipynb  GPU training on Colab
tests/                     engine, materials, pipeline (fake detector), dataset tools, DB
```

## Honest limitations (say these in interviews)

- **Overflow is a geometric rule, not a learned class.** A bin counts as overflowing when enough
  waste sits just outside its box. That depends on the labelling convention (bin box = the
  container only) and on the camera angle, and `bin_pad`, `min_overflow_items`, `min_items`,
  `min_area` and `roi` must be tuned per camera. It does not measure how full a closed bin is.
- If bins are not detected in a frame, all waste there is treated as loose and can raise a
  `garbage_dump` alert, so check bin detection quality, not only waste detection.
- **Area share is not weight or volume.** The mix is a share of visible area, and items
  hidden under other items are not seen.
- **Material from a photo is hard.** Plastic / glass / metal are easily confused, especially at
  a distance; expect weaker per-class results than overall mAP.
- **Domain gap.** Quality depends on how close your training images are to your real camera.
- The severity score (area x persistence) is not calibrated.
- With few clips the event metrics are indicative, not statistically strong - report how many
  clips and incidents you used.

## Interview cheat-sheet

- **Why detect items instead of classifying whole images?** One image contains several
  materials; boxes give counts, locations and a composition.
- **Why a sliding window and two thresholds?** Tolerates missed detections and prevents an alert
  flapping around one cut-off (hysteresis).
- **Why 1 FPS?** Waste does not move; far less compute, no information loss.
- **Why a geometric overflow rule instead of an `overflowing` class?** It needs only `bin` and
  material labels (no subjective "is it overflowing?" labelling), it is explainable ("2 items
  outside the bin box"), and the same waste boxes give both the alert and the material mix.
- **Why merge datasets through a class map?** Every dataset names classes differently; an
  explicit, reviewable mapping avoids silent label errors.
- **Why split by source?** Neighbouring frames are near-duplicates; random splits leak.
- **Why event-level metrics?** 0.85 mAP can still mean dozens of false alerts a day.

## Resume bullet (use YOUR measured numbers)

> Built a computer-vision waste monitoring system: fine-tuned YOLOv8 to detect bins and sort
> waste by material (plastic, metal, glass, paper, cardboard, organic) on a merged dataset of **N**
> images (mAP@50 = **X**), with geometric rules that raise overflowing-bin and illegal-dump
> alerts and a temporal alert engine that cut false/duplicate alerts from
> **Y** to **Z** per hour on **K** test clips; logged alerts with waste composition to SQLite and
> built a Streamlit dashboard with map and trends.
