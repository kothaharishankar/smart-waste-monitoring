"""Streamlit dashboard.   Run from the project root:

    streamlit run app/dashboard.py
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src import db
from src.config import get_camera, load_config, resolve
from src.segregate import analyse_image, average_composition, dominant, parse_composition, shares

st.set_page_config(page_title="Smart Waste Monitoring", page_icon="♻️", layout="wide")
cfg = load_config()
db_path = Path(os.environ.get("WASTE_DB") or resolve(cfg["paths"]["db"]))

st.sidebar.header("Settings")
st.sidebar.caption(f"Database: `{db_path}`")
if st.sidebar.button("Refresh"):
    st.rerun()


@st.cache_resource
def get_detector(weights, conf, imgsz):
    from src.detector import Detector
    return Detector(weights, conf, imgsz)


def load_detector():
    m = cfg["model"]
    try:
        return get_detector(str(resolve(m["weights"])), m["conf"], m["imgsz"])
    except FileNotFoundError as e:
        st.error(str(e))
        return None


def bar(series, container=st, empty="Nothing to show yet."):
    """st.bar_chart that tolerates empty data."""
    if series is None or len(series) == 0:
        container.caption(empty)
    else:
        container.bar_chart(series)


# ---------------------------------------------------------------- load events
df = db.fetch_events(db_path)
df["start"] = pd.to_datetime(df["start_ts"])
df["end"] = pd.to_datetime(df["end_ts"])
df["duration_min"] = (df["end"] - df["start"]).dt.total_seconds() / 60
df["status"] = np.where(df["end"].isna(), "ongoing", "cleared")
df["comp"] = df["composition"].apply(parse_composition)
df["main_material"] = df["comp"].apply(dominant)

st.title("♻️ Smart Waste Monitoring")
st.caption("Detects waste from camera feeds, sorts it by material and raises alerts when it piles up")

if df.empty:
    st.info("No alerts yet. Open **Analyse images** to see material detection on a photo, or run "
            "`python -m src.pipeline --video data/videos/<clip>.mp4` and press Refresh.")

# ------------------------------------------------------------------- filters
cams = sorted(df["camera_id"].dropna().unique())
types = sorted(df["type"].dropna().unique())
fc1, fc2, fc3 = st.columns(3)
sel_cams = fc1.multiselect("Camera", cams, default=cams)
sel_types = fc2.multiselect("Alert type", types, default=types)
sel_status = fc3.radio("Status", ["All", "ongoing", "cleared"], horizontal=True)

view = df[df["camera_id"].isin(sel_cams) & df["type"].isin(sel_types)]
if sel_status != "All":
    view = view[view["status"] == sel_status]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total alerts", len(view))
k2.metric("Ongoing", int((view["status"] == "ongoing").sum()))
k3.metric("High severity", int((view["severity_label"] == "high").sum()))
avg = view["duration_min"].mean()
k4.metric("Avg. time to clear (min)", "-" if pd.isna(avg) else f"{avg:.1f}")

tab_alerts, tab_map, tab_trends, tab_img, tab_vid = st.tabs(
    ["Alerts", "Map", "Trends", "Analyse images", "Analyse a video"])

# -------------------------------------------------------------------- alerts
with tab_alerts:
    cols = ["id", "camera_name", "type", "main_material", "severity_label", "status",
            "start_ts", "end_ts", "duration_min"]
    st.dataframe(view[cols].round({"duration_min": 1}), hide_index=True)
    if not view.empty:
        labels = {int(r.id): f"#{int(r.id)} · {r.camera_id} · {r.type} · {r.start_ts}"
                  for r in view.itertuples()}
        chosen = st.selectbox("Inspect an alert", list(labels), format_func=lambda i: labels[i])
        row = view.loc[view["id"] == chosen].iloc[0]
        left, right = st.columns([2, 1])
        snap = row["snapshot_path"]
        snap_path = resolve(snap) if isinstance(snap, str) else None
        if snap_path is not None and snap_path.exists():
            left.image(str(snap_path), caption=labels[chosen])
        else:
            left.caption("Snapshot file not found on disk.")
        right.subheader("Waste mix")
        if row["comp"]:
            bar(pd.Series(row["comp"], name="share of waste area"), right)
        else:
            right.caption("No material breakdown stored for this alert.")

# ----------------------------------------------------------------------- map
with tab_map:
    cam_df = db.fetch_cameras(db_path)
    if cam_df.empty:
        st.info("No cameras in the database yet.")
    else:
        try:
            import folium
            from streamlit_folium import st_folium

            m = folium.Map(location=[cam_df["lat"].mean(), cam_df["lon"].mean()], zoom_start=14)
            for c in cam_df.itertuples():
                ev = df[df["camera_id"] == c.id]
                ongoing = int((ev["status"] == "ongoing").sum())
                folium.Marker(
                    [c.lat, c.lon],
                    popup=f"{c.name}<br>Total alerts: {len(ev)}<br>Ongoing: {ongoing}",
                    tooltip=c.name,
                    icon=folium.Icon(color="red" if ongoing else "green", icon="trash", prefix="fa"),
                ).add_to(m)
            st_folium(m, height=450, width=900, returned_objects=[])
            st.caption("Red = camera has an ongoing alert. Edit coordinates in configs/params.yaml.")
        except ImportError:
            st.warning("Install `folium` and `streamlit-folium` to see the map.")
            st.map(cam_df.rename(columns={"lon": "longitude", "lat": "latitude"}))

# -------------------------------------------------------------------- trends
with tab_trends:
    if view.empty:
        st.info("Nothing to chart yet.")
    else:
        st.subheader("Waste composition across alerts")
        avg_comp = average_composition(view["comp"].tolist())
        if avg_comp:
            bar(pd.Series(avg_comp, name="average share of waste area"))
        else:
            st.caption("No material breakdown stored yet.")
        st.subheader("Alerts per day")
        bar(view.groupby(view["start"].dt.date).size().rename("alerts"))
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Main material per alert")
            bar(view["main_material"].value_counts())
        with c2:
            st.subheader("By severity")
            bar(view["severity_label"].value_counts())
        st.subheader("Alerts by hour of day")
        bar(view.groupby(view["start"].dt.hour).size().rename("alerts"))

# ------------------------------------------------------------- analyse images
with tab_img:
    st.write("Upload photos to see the model separate the waste by material "
             "(each material has its own colour).")
    ups = st.file_uploader("Images", type=["jpg", "jpeg", "png", "bmp", "webp"],
                           accept_multiple_files=True, key="imgs")
    if ups and st.button("Analyse images"):
        detector = load_detector()
        if detector is not None:
            for up in ups:
                img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    st.warning(f"Could not read {up.name}")
                    continue
                annotated, info = analyse_image(detector, img, cfg)
                st.subheader(up.name)
                a, b = st.columns([2, 1])
                a.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                if info.counts:
                    sh = shares(info.comp)
                    table = pd.DataFrame({
                        "material": list(info.counts),
                        "items": list(info.counts.values()),
                        "share of waste area": [f"{sh.get(k, 0):.0%}" for k in info.counts],
                    }).sort_values("items", ascending=False)
                    b.dataframe(table, hide_index=True)
                    bar(pd.Series(sh, name="share of waste area"), b)
                else:
                    b.info("No waste detected above the confidence threshold.")
                n_over = sum(1 for x in info.bins if x["overflowing"])
                b.write(f"Bins detected: **{len(info.bins)}** - overflowing: **{n_over}**")
                b.write("Alert: **" + (info.label if info.flagged else "none") + "**")

# ------------------------------------------------------------- analyse a video
with tab_vid:
    st.write("Upload a short clip. The model samples frames, the event engine decides whether "
             "waste is really piling up, and confirmed alerts are saved to the database.")
    cam_choice = st.selectbox("Camera", [c["id"] for c in cfg["cameras"]])
    up = st.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"], key="vid")
    if up is not None and st.button("Run detection"):
        detector = load_detector()
        if detector is not None:
            from src.pipeline import process_video

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix)
            tmp.write(up.read())
            tmp.close()
            bar = st.progress(0.0)
            conn = db.connect(resolve(cfg["paths"]["db"]))
            for c in cfg["cameras"]:
                db.upsert_camera(conn, c)
            try:
                res = process_video(tmp.name, detector, cfg, camera=get_camera(cfg, cam_choice),
                                    conn=conn, progress=bar.progress)
            finally:
                conn.close()
                os.unlink(tmp.name)
            st.success(f"Analysed {res.frames_sampled} frames - {len(res.events)} confirmed alert(s). "
                       "Press Refresh in the sidebar to see them in the Alerts tab.")
            for e in res.events:
                mix = ", ".join(f"{k} {v:.0%}" for k, v in e["composition"].items()) or "n/a"
                st.write(f"**{e['type']}** from {e['start_s']:.0f}s - severity **{e['severity_label']}** "
                         f"- mix: {mix}")
                p = resolve(e["snapshot"])
                if p.exists():
                    st.image(str(p))
