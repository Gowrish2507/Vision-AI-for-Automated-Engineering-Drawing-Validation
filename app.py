import streamlit as st
import numpy as np
import pandas as pd
import cv2
import pytesseract
from pytesseract import Output
import joblib
import os
import io
import re
import math
import json
import time
import hashlib
import inspect
import tempfile
import traceback
import html as pyhtml

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont
from fpdf import FPDF

# ---------------------------------------------------------
# PAGE CONFIG  (must be the first Streamlit call, only once)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Eng Drawing Validation",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# TESSERACT + PATHS
# ---------------------------------------------------------
_WIN_TESS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(_WIN_TESS):
    pytesseract.pytesseract.tesseract_cmd = _WIN_TESS

APP_DIR = os.path.dirname(os.path.abspath(__file__))
_LEGACY_DIR = r"C:\Users\CRESCENT DEV\Desktop\ENGDrawing Validation"
BASE_DIR = APP_DIR if os.path.isdir(os.path.join(APP_DIR, "models")) else _LEGACY_DIR
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
PAGE_DASH = "🏠 Dashboard"
PAGE_VAL = "📐 Validate Drawing"
PAGE_MAT = "🧱 Materials"
PAGE_REP = "📥 Export Report"
PAGE_HIS = "📜 History"
PAGE_ABT = "ℹ️ About"
PAGES = [PAGE_DASH, PAGE_VAL, PAGE_MAT, PAGE_REP, PAGE_HIS, PAGE_ABT]

UNIT_FACTORS = {"Metres (m)": 1.0, "Millimetres (mm)": 0.001, "Feet (ft)": 0.3048}
ROOM_COLS = ["Room", "Length (m)", "Width (m)"]

# Colour system  (violet + orange + rose + teal + amber + emerald)
VIOLET, ORANGE, ROSE, TEAL, AMBER, EMERALD, PINK = (
    "#7C3AED", "#F97316", "#E11D48", "#0D9488", "#F59E0B", "#10B981", "#DB2777")
INK = "#2B2140"
COLORWAY = [VIOLET, ORANGE, TEAL, PINK, AMBER, EMERALD, "#6366F1", ROSE]

STATUS_META = {
    "Valid": (EMERALD, TEAL, "✅", "Estimation is Valid"),
    "Minor Difference": (AMBER, ORANGE, "⚠️", "Minor Difference found"),
    "Invalid": (ROSE, PINK, "❌", "Estimation is Invalid"),
}

# Material take-off assumptions (all documented on the About page)
DOOR_AREA, WINDOW_AREA = 2.1, 1.44            # m² per opening
ROOM_OPENINGS = DOOR_AREA + WINDOW_AREA        # 1 door + 1 window per room
BRICK_VOL = 0.002                              # m³ per brick incl. mortar
PLASTER_T, BAGS_PER_M3 = 0.012, 8              # 12 mm plaster, ~8 bags / m³ mortar
TILE_AREA, TILE_WASTE = 0.36, 1.05             # 600x600 mm + 5 % wastage
PAINT_COVER, PAINT_COATS = 10.0, 2             # m² per litre per coat, 2 coats
COLUMN_SPACING = 4.0                           # m, rule-of-thumb grid

FEATURE_COLS = ["Component_Type_Encoded", "Detected_Length", "Detected_Area",
                "Detected_Quantity", "Estimated_Length", "Estimated_Area",
                "Estimated_Quantity", "Floor_Area", "Tolerance_Percentage"]

# ---------------------------------------------------------
# MODELS  (optional - app still validates with rule engine if missing)
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        return {
            "reg": joblib.load(os.path.join(MODEL_DIR, "rf_regressor.pkl")),
            "clf": joblib.load(os.path.join(MODEL_DIR, "rf_classifier.pkl")),
            "le_comp": joblib.load(os.path.join(MODEL_DIR, "le_component.pkl")),
            "le_stat": joblib.load(os.path.join(MODEL_DIR, "le_status.pkl")),
            "error": None,
        }
    except Exception as e:  # noqa
        return {"reg": None, "clf": None, "le_comp": None, "le_stat": None, "error": str(e)}


MODELS = load_models()


@st.cache_resource(show_spinner=False)
def tesseract_ok():
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


# ---------------------------------------------------------
# SESSION STATE  (single source of truth shared by all pages)
# ---------------------------------------------------------
def empty_rooms():
    return pd.DataFrame({"Room": pd.Series(dtype="object"),
                         "Length (m)": pd.Series(dtype="float64"),
                         "Width (m)": pd.Series(dtype="float64")})


DEFAULTS = {
    "page": PAGE_DASH,
    # calibration
    "calib_mode": "Auto (OCR)", "unit": "Metres (m)", "man_w": 10.0, "man_d": 10.0,
    # estimations
    "est_area": 100.0, "est_wall": 50.0, "wall_h": 3.0,
    "est_doors": 4, "est_windows": 6, "est_columns": 6,
    "est_cost": 50000, "tolerance": 10,
    # construction assumptions
    "wall_thk": 0.23, "slab_thk": 0.125, "col_size": 0.30,
    # editable placeholder rates (INR) - set your local schedule of rates
    "rate_brick": 8.0, "rate_cement": 400.0, "rate_concrete": 6500.0,
    "rate_tile": 150.0, "rate_paint": 350.0,
    # history filters
    "hist_q": "", "hist_f": ["Valid", "Minor Difference", "Invalid"],
    # drawing state
    "img_bytes": None, "img_name": None, "img_sig": None, "last_up_sig": None,
    "ocr": None, "up_ver": 0, "visit": 0, "_last_page": None,
    "history": [], "last_saved_sig": None,
}
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
if "rooms_df" not in st.session_state:
    st.session_state.rooms_df = empty_rooms()
    st.session_state.rooms_snapshot = empty_rooms()

# Keep widget values alive while the user moves between pages
PERSIST = ["calib_mode", "unit", "man_w", "man_d", "est_area", "est_wall", "wall_h",
           "est_doors", "est_windows", "est_columns", "est_cost", "tolerance",
           "wall_thk", "slab_thk", "col_size", "rate_brick", "rate_cement",
           "rate_concrete", "rate_tile", "rate_paint", "hist_q", "hist_f"]
for _k in PERSIST:
    st.session_state[_k] = st.session_state[_k]

ss = st.session_state

# fresh editor widget every time a page is (re)visited -> edits are never lost
if ss._last_page != ss.page:
    ss.visit += 1
    ss._last_page = ss.page
    ss.rooms_snapshot = ss.rooms_df.copy()


def goto(page):
    ss.page = page


# ---------------------------------------------------------
# SMALL UTILITIES
# ---------------------------------------------------------
esc = pyhtml.escape


def inr(x):
    x = float(x)
    neg = x < 0
    s = str(int(abs(round(x))))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        s = head + "," + tail
    return ("-" if neg else "") + "₹" + s


def html(s):
    """Render raw HTML safely (newlines collapsed so Markdown never treats it as code)."""
    st.markdown(" ".join(l.strip() for l in s.strip().splitlines() if l.strip()),
                unsafe_allow_html=True)


def calc_dev(detected, estimated):
    if detected == 0:
        return 0.0 if estimated == 0 else 100.0
    return round(abs(detected - estimated) / detected * 100, 2)


def comp_result(dev, tol):
    if dev <= tol:
        return "✅ Within tolerance"
    if dev <= tol * 1.5:
        return "⚠️ Slightly outside"
    return "❌ Outside tolerance"


_PLOT_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


def plot(fig, key):
    cfg = {"displayModeBar": False}
    if _PLOT_HAS_WIDTH:
        st.plotly_chart(fig, width="stretch", key=key, config=cfg)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key, config=cfg)


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def show_image(img, caption=None):
    for kwargs in ({"width": "stretch"}, {"use_container_width": True}, {"use_column_width": True}):
        try:
            st.image(img, caption=caption, **kwargs)
            return
        except Exception:
            continue


def editor(df, key, cfg):
    try:
        return st.data_editor(df, key=key, num_rows="dynamic", hide_index=True,
                              column_config=cfg, width="stretch")
    except Exception:
        return st.data_editor(df, key=key, num_rows="dynamic", hide_index=True,
                              column_config=cfg, use_container_width=True)


# ---------------------------------------------------------
# STYLING  (theme: blueprint + construction palette)
# ---------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=JetBrains+Mono:wght@600;700&display=swap');
:root{--violet:#7C3AED;--orange:#F97316;--rose:#E11D48;--teal:#0D9488;--amber:#F59E0B;--emerald:#10B981;--pink:#DB2777;--ink:#2B2140;--muted:#6B5F80;}
.stApp{font-family:'Poppins','Segoe UI',sans-serif;color:var(--ink);
background-image:linear-gradient(rgba(124,58,237,.045) 1px,transparent 1px),linear-gradient(90deg,rgba(124,58,237,.045) 1px,transparent 1px),linear-gradient(135deg,#FFF7ED 0%,#FDF2F8 48%,#ECFEFF 100%);
background-size:34px 34px,34px 34px,100% 100%;background-attachment:fixed;}
header[data-testid="stHeader"]{background:transparent;}
footer{visibility:hidden;}
.block-container{padding-top:1.1rem;padding-bottom:3rem;max-width:1280px;}
h1,h2,h3,h4{color:var(--ink);letter-spacing:-.01em;}
@keyframes fadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes gradShift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes sweep{0%{left:-60%}55%,100%{left:130%}}
@keyframes draw{to{stroke-dashoffset:0}}
@keyframes blink{0%,35%{opacity:0}65%,100%{opacity:1}}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(167,243,208,.75)}70%{box-shadow:0 0 0 9px rgba(167,243,208,0)}100%{box-shadow:0 0 0 0 rgba(167,243,208,0)}}
@keyframes pop{0%{transform:scale(.93);opacity:0}60%{transform:scale(1.025);opacity:1}100%{transform:scale(1)}}
@keyframes floaty{0%,100%{transform:translateY(0)}50%{transform:translateY(-7px)}}
@keyframes grow{from{width:0}}
@keyframes ring{0%{box-shadow:0 0 0 0 rgba(249,115,22,.55)}70%{box-shadow:0 0 0 10px rgba(249,115,22,0)}100%{box-shadow:0 0 0 0 rgba(249,115,22,0)}}
/* ---------- sidebar ---------- */
section[data-testid="stSidebar"]>div:first-child{background:linear-gradient(185deg,#3B0764 0%,#6D28D9 40%,#BE185D 82%,#EA580C 100%);}
section[data-testid="stSidebar"] *{color:#fff;}
section[data-testid="stSidebar"] [role="radiogroup"]{gap:.4rem;}
section[data-testid="stSidebar"] [role="radiogroup"] label{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.15);border-radius:13px;padding:.55rem .85rem;width:100%;transition:all .25s ease;cursor:pointer;}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:rgba(255,255,255,.22);transform:translateX(5px);}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:linear-gradient(90deg,rgba(255,255,255,.36),rgba(255,255,255,.1));border-color:rgba(255,255,255,.55);box-shadow:0 8px 18px rgba(0,0,0,.2);font-weight:700;}
section[data-testid="stSidebar"] [role="radiogroup"] label>div:first-child{display:none;}
section[data-testid="stSidebar"] .stButton>button{background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.32);box-shadow:none;}
section[data-testid="stSidebar"] .stButton>button:hover{background:rgba(255,255,255,.3);}
.logo{display:flex;align-items:center;gap:12px;margin:.2rem 0 .8rem;}
.logo-ic{width:50px;height:50px;border-radius:16px;display:flex;align-items:center;justify-content:center;font-size:1.7rem;background:rgba(255,255,255,.18);border:1px solid rgba(255,255,255,.4);animation:floaty 4s ease-in-out infinite;}
.logo-t{font-weight:800;font-size:1.15rem;line-height:1.1;}
.logo-s{font-size:.72rem;opacity:.85;}
.sb-card{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.22);border-radius:14px;padding:10px 12px;margin:.45rem 0;font-size:.8rem;}
.sb-row{display:flex;justify-content:space-between;gap:8px;padding:2px 0;}
/* ---------- hero ---------- */
.hero{position:relative;overflow:hidden;border-radius:26px;padding:30px 36px;color:#fff;display:flex;align-items:center;justify-content:space-between;gap:20px;margin-bottom:1.1rem;
background:linear-gradient(120deg,#6D28D9,#DB2777,#F97316,#0D9488,#6D28D9);background-size:300% 300%;
animation:gradShift 14s ease infinite,fadeUp .7s ease both;box-shadow:0 18px 44px rgba(109,40,217,.28);}
.hero::before{content:"";position:absolute;inset:0;background-image:linear-gradient(rgba(255,255,255,.13) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.13) 1px,transparent 1px);background-size:28px 28px;-webkit-mask-image:linear-gradient(90deg,transparent,#000 75%);mask-image:linear-gradient(90deg,transparent,#000 75%);}
.hero::after{content:"";position:absolute;top:0;left:-60%;width:38%;height:100%;background:linear-gradient(100deg,transparent,rgba(255,255,255,.3),transparent);transform:skewX(-20deg);animation:sweep 6s ease-in-out infinite;}
.hero-l{position:relative;z-index:2;max-width:680px;}
.hero h1{color:#fff;margin:0 0 .2rem;font-size:2.15rem;font-weight:800;line-height:1.12;text-shadow:0 2px 14px rgba(0,0,0,.2);padding:0;}
.hero p{margin:.25rem 0 .95rem;color:rgba(255,255,255,.93);font-size:1rem;}
.hero-bar{height:4px;width:90px;border-radius:4px;background:#fff;margin:.35rem 0 .6rem;animation:grow 1.4s ease both;}
.chip{display:inline-flex;align-items:center;gap:7px;background:rgba(255,255,255,.18);border:1px solid rgba(255,255,255,.38);backdrop-filter:blur(6px);padding:5px 13px;border-radius:999px;font-size:.78rem;font-weight:600;margin:0 7px 7px 0;}
.dot{width:9px;height:9px;border-radius:50%;background:#A7F3D0;animation:pulse 1.9s infinite;}
.dot.off{background:#FDE68A;animation:none;}
.dot.bad{background:#FECACA;animation:none;}
.hero-r{position:relative;z-index:2;flex:0 0 270px;animation:floaty 6s ease-in-out infinite;}
.draw{fill:none;stroke:#fff;stroke-width:2.6;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:100;stroke-dashoffset:100;animation:draw 4.5s ease-in-out infinite alternate;}
.dimtxt{fill:#fff;font:700 12px 'JetBrains Mono',monospace;animation:blink 4.5s ease-in-out infinite alternate;}
@media(max-width:950px){.hero-r{display:none}.hero h1{font-size:1.6rem}.hero{padding:22px}}
/* ---------- section titles / cards ---------- */
.sec{display:flex;align-items:center;gap:12px;margin:1.5rem 0 .65rem;animation:fadeUp .5s ease both;}
.sec-ic{width:42px;height:42px;border-radius:13px;display:flex;align-items:center;justify-content:center;font-size:1.3rem;background:linear-gradient(135deg,#EDE9FE,#FCE7F3);box-shadow:0 6px 14px rgba(124,58,237,.14);}
.sec-t{font-size:1.22rem;font-weight:700;color:var(--ink);line-height:1.2;}
.sec-s{font-size:.84rem;color:var(--muted);}
.card{background:rgba(255,255,255,.82);backdrop-filter:blur(8px);border-radius:18px;padding:18px 22px;border:1px solid rgba(124,58,237,.12);box-shadow:0 8px 22px rgba(60,20,120,.07);animation:fadeUp .55s ease both;}
.empty{text-align:center;padding:38px 24px;background:rgba(255,255,255,.78);border:2px dashed rgba(124,58,237,.35);border-radius:22px;animation:fadeUp .6s ease both;}
.empty .ei{font-size:3rem;animation:floaty 3.5s ease-in-out infinite;display:inline-block;}
.empty h3{margin:.4rem 0 .2rem;}
.empty p{color:var(--muted);margin:0;}
/* ---------- KPI ---------- */
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:.4rem 0 1rem;}
.kpi{position:relative;display:flex;gap:13px;align-items:center;background:rgba(255,255,255,.86);backdrop-filter:blur(8px);border-radius:18px;padding:16px 16px 16px 22px;box-shadow:0 8px 22px rgba(60,20,120,.08);border:1px solid rgba(124,58,237,.1);overflow:hidden;animation:fadeUp .55s ease both;transition:transform .25s ease,box-shadow .25s ease;}
.kpi::before{content:"";position:absolute;left:0;top:0;bottom:0;width:6px;background:var(--c);}
.kpi:hover{transform:translateY(-5px);box-shadow:0 16px 32px rgba(60,20,120,.17);}
.kpi-ic{flex:0 0 auto;font-size:1.6rem;width:50px;height:50px;border-radius:15px;display:flex;align-items:center;justify-content:center;background:rgba(124,58,237,.1);background:color-mix(in srgb,var(--c) 15%,white);}
.kpi-l{font-size:.7rem;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);font-weight:600;}
.kpi-v{font-size:1.4rem;font-weight:800;line-height:1.15;color:var(--ink);word-break:break-word;}
.kpi-s{font-size:.76rem;color:var(--muted);}
/* ---------- stepper ---------- */
.steps{display:flex;gap:10px;flex-wrap:wrap;margin:.3rem 0 1rem;}
.step{flex:1 1 160px;display:flex;align-items:center;gap:11px;background:rgba(255,255,255,.85);border-radius:15px;padding:11px 13px;border:1px solid rgba(124,58,237,.12);animation:fadeUp .5s ease both;}
.step-n{flex:0 0 auto;width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;background:#EDE9FE;color:#6D28D9;}
.step.done .step-n{background:linear-gradient(135deg,#10B981,#0D9488);color:#fff;}
.step.now .step-n{background:linear-gradient(135deg,#F97316,#DB2777);color:#fff;animation:ring 1.8s infinite;}
.step-t{font-size:.83rem;font-weight:700;line-height:1.15;}
.step-s{font-size:.7rem;color:var(--muted);}
/* ---------- result banner ---------- */
.banner{display:flex;align-items:center;gap:18px;border-radius:22px;padding:20px 26px;color:#fff;background:linear-gradient(120deg,var(--c1),var(--c2));box-shadow:0 14px 34px rgba(0,0,0,.15);animation:pop .6s ease both;margin:.6rem 0 1rem;}
.banner .ic{font-size:2.6rem;}
.banner h3{color:#fff;margin:0;font-size:1.4rem;}
.banner p{margin:.15rem 0 0;opacity:.95;font-size:.92rem;}
.banner .big{margin-left:auto;text-align:right;font-size:2.2rem;font-weight:800;line-height:1;}
.banner .big small{display:block;font-size:.72rem;font-weight:600;opacity:.9;margin-top:4px;}
/* ---------- room cards ---------- */
.rooms{display:grid;grid-template-columns:repeat(auto-fill,minmax(235px,1fr));gap:14px;margin:.4rem 0 1rem;}
.room{background:rgba(255,255,255,.9);border-radius:18px;padding:14px 16px 12px;border-top:5px solid var(--c);box-shadow:0 8px 20px rgba(60,20,120,.09);animation:fadeUp .5s ease both;transition:transform .25s ease,box-shadow .25s ease;}
.room:hover{transform:translateY(-6px);box-shadow:0 18px 34px rgba(60,20,120,.18);}
.room-h{display:flex;justify-content:space-between;align-items:baseline;gap:8px;}
.room-n{font-weight:800;color:var(--ink);}
.room-a{font-weight:700;color:var(--c);}
.room-d{font-size:.78rem;color:var(--muted);margin:.15rem 0 .65rem;}
.room-m{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;text-align:center;}
.room-m div{background:rgba(124,58,237,.07);border-radius:11px;padding:7px 2px;}
.room-m b{display:block;font-size:.95rem;color:var(--ink);}
.room-m span{font-size:.62rem;text-transform:uppercase;color:var(--muted);letter-spacing:.05em;}
.bar{height:5px;border-radius:5px;background:#F1EAFB;overflow:hidden;margin-top:10px;}
.bar i{display:block;height:100%;background:var(--c);width:var(--w);border-radius:5px;animation:grow 1.1s ease both;}
/* ---------- callouts ---------- */
.co{display:flex;gap:12px;border-radius:15px;padding:12px 15px;margin:.45rem 0;background:rgba(255,255,255,.92);border-left:6px solid var(--c);box-shadow:0 4px 14px rgba(0,0,0,.06);animation:fadeUp .45s ease both;}
.co .i{font-size:1.25rem;}
.co b{display:block;color:var(--ink);}
.co span{color:var(--muted);font-size:.9rem;}
.risk{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;margin:.4rem 0 .8rem;}
.risk>div{border-radius:18px;padding:15px 18px;background:rgba(255,255,255,.9);border-top:5px solid var(--c);box-shadow:0 8px 20px rgba(60,20,120,.08);animation:fadeUp .5s ease both;}
.risk small{display:block;text-transform:uppercase;letter-spacing:.08em;font-size:.68rem;color:var(--muted);font-weight:700;}
.risk b{font-size:1.05rem;}
.legend span{display:inline-block;margin-right:14px;font-size:.8rem;color:var(--muted);}
.legend i{display:inline-block;width:12px;height:12px;border-radius:4px;margin-right:6px;vertical-align:-1px;}
.foot{text-align:center;color:var(--muted);font-size:.78rem;margin-top:2.6rem;padding-top:1rem;border-top:1px dashed rgba(124,58,237,.28);}
/* ---------- native widgets ---------- */
.stButton>button,.stDownloadButton>button{border:0;border-radius:13px;font-weight:700;padding:.62rem 1.15rem;color:#fff;background:linear-gradient(90deg,#7C3AED,#DB2777,#F97316);background-size:200% 100%;background-position:0 0;box-shadow:0 8px 18px rgba(124,58,237,.25);transition:all .3s ease;width:100%;}
.stButton>button p,.stDownloadButton>button p{color:#fff;font-weight:700;}
.stButton>button:hover,.stDownloadButton>button:hover{background-position:100% 0;transform:translateY(-3px);box-shadow:0 14px 26px rgba(219,39,119,.32);color:#fff;border:0;}
.stButton>button:active,.stDownloadButton>button:active{transform:translateY(0) scale(.98);}
div[data-baseweb="input"],div[data-baseweb="select"]>div,div[data-baseweb="base-input"]{border-radius:12px !important;}
div[data-testid="stNumberInput"] input{font-weight:600;}
[data-testid="stFileUploader"] section{border:2px dashed rgba(124,58,237,.5);border-radius:18px;background:rgba(255,255,255,.75);transition:all .25s ease;}
[data-testid="stFileUploader"] section:hover{border-color:#DB2777;background:rgba(253,242,248,.95);transform:scale(1.01);}
[data-testid="stPlotlyChart"]{background:rgba(255,255,255,.82);border-radius:18px;padding:8px;border:1px solid rgba(124,58,237,.1);box-shadow:0 8px 22px rgba(60,20,120,.07);animation:fadeUp .6s ease both;}
[data-testid="stDataFrame"],[data-testid="stDataEditor"]{border-radius:14px;overflow:hidden;box-shadow:0 6px 18px rgba(60,20,120,.07);}
[data-testid="stExpander"]{border-radius:15px;background:rgba(255,255,255,.75);border:1px solid rgba(124,58,237,.16);}
[data-testid="stImage"] img{border-radius:14px;box-shadow:0 8px 22px rgba(60,20,120,.12);}
button[data-baseweb="tab"]{font-weight:600;}
[data-baseweb="tab-highlight"]{background:linear-gradient(90deg,#7C3AED,#DB2777,#F97316) !important;height:3px !important;}
@media (prefers-reduced-motion:reduce){*{animation:none !important;transition:none !important;}}
"""

HERO_SVG = """
<svg viewBox="0 0 300 205" width="100%" xmlns="http://www.w3.org/2000/svg">
<rect class="draw" pathLength="100" x="34" y="36" width="232" height="140"/>
<path class="draw" pathLength="100" style="animation-delay:.5s" d="M142 36 V112 H34"/>
<path class="draw" pathLength="100" style="animation-delay:.9s" d="M196 112 H266 M196 112 V176"/>
<path class="draw" pathLength="100" style="animation-delay:1.3s" d="M96 112 a22 22 0 0 1 22 22"/>
<path class="draw" pathLength="100" style="animation-delay:1.6s" d="M196 150 a22 22 0 0 0 -22 22"/>
<path class="draw" pathLength="100" style="animation-delay:1.9s;stroke-width:1.4" d="M34 22 H266 M34 17 V27 M266 17 V27"/>
<path class="draw" pathLength="100" style="animation-delay:2.1s;stroke-width:1.4" d="M282 36 V176 M277 36 H287 M277 176 H287"/>
<text class="dimtxt" x="150" y="13" text-anchor="middle">15.00</text>
<text class="dimtxt" x="270" y="196" text-anchor="end">12.50</text>
</svg>
"""


def inject_css():
    html("<style>" + CSS + "</style>")


# ---------------------------------------------------------
# REUSABLE UI BLOCKS
# ---------------------------------------------------------
def hero(title, subtitle, section_name):
    ocr_ok, mdl_ok = tesseract_ok(), MODELS["error"] is None
    chips = [
        ("", f"📍 {section_name}"),
        ("" if mdl_ok else "off", "ML models ready" if mdl_ok else "ML models missing · rule-based mode"),
        ("" if ocr_ok else "bad", "OCR engine ready" if ocr_ok else "OCR engine not found"),
        ("" if ss.img_bytes else "off", f"Drawing: {esc(str(ss.img_name))}" if ss.img_bytes else "No drawing loaded"),
    ]
    chips_html = "".join(f'<span class="chip"><span class="dot {c}"></span>{t}</span>' for c, t in chips)
    html(f'<div class="hero"><div class="hero-l"><h1>📐 {esc(title)}</h1><div class="hero-bar"></div>'
         f'<p>{esc(subtitle)}</p>{chips_html}</div><div class="hero-r">{HERO_SVG}</div></div>')


def section(icon, title, sub=""):
    html(f'<div class="sec"><div class="sec-ic">{icon}</div><div><div class="sec-t">{esc(title)}</div>'
         f'<div class="sec-s">{esc(sub)}</div></div></div>')


def kpi(icon, label, value, sub="", color=VIOLET, delay=0):
    return (f'<div class="kpi" style="--c:{color};animation-delay:{delay}ms"><div class="kpi-ic">{icon}</div>'
            f'<div><div class="kpi-l">{esc(label)}</div><div class="kpi-v">{value}</div>'
            f'<div class="kpi-s">{sub}</div></div></div>')


def kpi_grid(items):
    html('<div class="kpi-grid">' + "".join(items) + "</div>")


def callout(kind, title, text):
    color, icon = {"ok": (EMERALD, "✅"), "warn": (AMBER, "⚠️"), "high": (ROSE, "❌"),
                   "info": (VIOLET, "ℹ️")}.get(kind, (VIOLET, "ℹ️"))
    html(f'<div class="co" style="--c:{color}"><div class="i">{icon}</div><div><b>{esc(title)}</b>'
         f'<span>{esc(text)}</span></div></div>')


def empty_state(icon, title, text):
    html(f'<div class="empty"><div class="ei">{icon}</div><h3>{esc(title)}</h3><p>{esc(text)}</p></div>')


def footer():
    html('<div class="foot">Engineering Drawing Estimation Validation System · Streamlit · OpenCV · '
         'Tesseract OCR · scikit-learn · Plotly · Final-year project</div>')


def status_banner(A):
    c1, c2, icon, title = STATUS_META[A["status"]]
    html(f'<div class="banner" style="--c1:{c1};--c2:{c2}"><div class="ic">{icon}</div><div><h3>{title}</h3>'
         f'<p>Largest component deviation is {A["overall_dev"]:.2f}% against an allowed tolerance of '
         f'{A["tol"]:.0f}% (worst: {esc(A["worst"])}).</p></div>'
         f'<div class="big">{A["overall_dev"]:.1f}%<small>of {A["tol"]:.0f}% allowed</small></div></div>')


def room_cards(rm_df, unsized):
    if rm_df.empty:
        callout("info", "No room sizes available yet",
                "Add or correct rooms in the ‘Detected Rooms’ table and the cards, graphs and materials update instantly.")
        return
    total = float(rm_df["Area (m²)"].sum())
    cards = []
    for i, r in rm_df.reset_index(drop=True).iterrows():
        c = COLORWAY[i % len(COLORWAY)]
        share = (r["Area (m²)"] / total * 100) if total else 0
        cards.append(
            f'<div class="room" style="--c:{c};animation-delay:{i * 70}ms">'
            f'<div class="room-h"><span class="room-n">{esc(str(r["Room"]))}</span>'
            f'<span class="room-a">{r["Area (m²)"]:.2f} m²</span></div>'
            f'<div class="room-d">{r["Length (m)"]:.2f} × {r["Width (m)"]:.2f} m · {share:.0f}% of room area</div>'
            f'<div class="room-m"><div><b>{int(r["Tiles (pcs)"]):,}</b><span>Tiles</span></div>'
            f'<div><b>{r["Paint (L)"]:.1f} L</b><span>Paint</span></div>'
            f'<div><b>{r["Cement (bags)"]:.1f}</b><span>Cement bags</span></div></div>'
            f'<div class="bar" style="--c:{c};--w:{share:.0f}%"><i></i></div></div>')
    html('<div class="rooms">' + "".join(cards) + "</div>")
    if unsized:
        callout("warn", "Some rooms have no size",
                "Size not read for: " + ", ".join(unsized) + ". Fill Length/Width in the rooms table to include them.")


# ---------------------------------------------------------
# PLOTLY FIGURES
# ---------------------------------------------------------
def style_fig(fig, height=360, title=None):
    fig.update_layout(
        height=height, margin=dict(l=12, r=12, t=58 if title else 40, b=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.0)",
        font=dict(family="Poppins, Segoe UI, sans-serif", color=INK, size=12.5),
        legend=dict(orientation="h", y=1.14, x=0), colorway=COLORWAY,
        title=dict(text=title, x=0.01, font=dict(size=16)) if title else None,
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(gridcolor="rgba(124,58,237,.09)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(124,58,237,.09)", zeroline=False)
    return fig


def fig_gauge(overall, tol):
    mx = max(overall * 1.25, tol * 3, 10)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=overall,
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": "Overall deviation (worst component)", "font": {"size": 14}},
        gauge={"axis": {"range": [0, mx]}, "bar": {"color": VIOLET},
               "steps": [{"range": [0, tol], "color": "#A7F3D0"},
                         {"range": [tol, tol * 1.5], "color": "#FDE68A"},
                         {"range": [tol * 1.5, mx], "color": "#FECDD3"}],
               "threshold": {"line": {"color": ROSE, "width": 4}, "thickness": .85, "value": tol}}))
    return style_fig(fig, 300).update_layout(margin=dict(l=25, r=25, t=60, b=10))


def fig_compare(comp_df):
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=.1,
                        subplot_titles=("Area (m²) & wall length (m)", "Doors, windows & columns (nos)"))
    for i, sub in enumerate((comp_df.iloc[:2], comp_df.iloc[2:]), start=1):
        fig.add_trace(go.Bar(name="Your estimate", x=sub["Component"], y=sub["Your Estimate"], marker_color=VIOLET,
                             texttemplate="%{y:,.1f}", textposition="outside", cliponaxis=False,
                             showlegend=(i == 1), legendgroup="e"), row=1, col=i)
        fig.add_trace(go.Bar(name="Drawing (detected)", x=sub["Component"], y=sub["Drawing"], marker_color=ORANGE,
                             texttemplate="%{y:,.1f}", textposition="outside", cliponaxis=False,
                             showlegend=(i == 1), legendgroup="d"), row=1, col=i)
    fig.update_layout(barmode="group")
    return style_fig(fig, 340)


def fig_dev(comp_df, tol):
    devs = comp_df["Deviation (%)"].tolist()
    colors = [EMERALD if d <= tol else AMBER if d <= tol * 1.5 else ROSE for d in devs]
    fig = go.Figure(go.Bar(x=comp_df["Component"], y=devs, marker_color=colors,
                           text=[f"{d:.1f}%" for d in devs], textposition="outside", cliponaxis=False,
                           customdata=comp_df["Result"],
                           hovertemplate="%{x}<br>%{y:.2f}% · %{customdata}<extra></extra>"))
    fig.add_hline(y=tol, line_dash="dash", line_color=ROSE, annotation_text=f"Tolerance {tol:.0f}%",
                  annotation_position="top left")
    fig.update_yaxes(range=[0, max(max(devs), tol * 1.6) * 1.25], title="Deviation (%)")
    return style_fig(fig, 340, "Deviation per component vs tolerance")


def fig_room_materials(rm):
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=.08,
                        subplot_titles=("Floor tiles (pcs)", "Wall paint (litres)", "Ceiling plaster cement (bags)"))
    cols = [COLORWAY[i % len(COLORWAY)] for i in range(len(rm))]
    for k, (col, fmt) in enumerate([("Tiles (pcs)", "%{y:,.0f}"), ("Paint (L)", "%{y:.1f}"),
                                    ("Cement (bags)", "%{y:.1f}")], start=1):
        fig.add_trace(go.Bar(x=rm["Room"], y=rm[col], marker_color=cols, texttemplate=fmt,
                             textposition="outside", cliponaxis=False, showlegend=False), row=1, col=k)
    fig.update_xaxes(tickangle=-30)
    return style_fig(fig, 360, "Material demand per room")


def fig_room_share(rm):
    fig = go.Figure(go.Pie(labels=rm["Room"], values=rm["Area (m²)"], hole=.58, sort=False,
                           marker=dict(colors=[COLORWAY[i % len(COLORWAY)] for i in range(len(rm))]),
                           textinfo="label+percent"))
    fig.add_annotation(text=f"<b>{rm['Area (m²)'].sum():.1f}</b><br>m²", showarrow=False, font=dict(size=16))
    return style_fig(fig, 340, "Room area share").update_layout(showlegend=False)


def fig_cost_donut(items):
    labels, vals = list(items.keys()), list(items.values())
    fig = go.Figure(go.Pie(labels=labels, values=vals, hole=.58, sort=False,
                           marker=dict(colors=COLORWAY[:len(labels)]), textinfo="label+percent"))
    fig.add_annotation(text=f"<b>{inr(sum(vals))}</b><br>materials", showarrow=False, font=dict(size=14))
    return style_fig(fig, 340, "Material cost split (drawing basis)").update_layout(showlegend=False)


def fig_global(glob_df):
    names = glob_df["Material"].tolist()
    fig = make_subplots(rows=1, cols=len(names), horizontal_spacing=.07, subplot_titles=names)
    for i, (_, r) in enumerate(glob_df.iterrows(), start=1):
        fig.add_trace(go.Bar(x=["Your estimate", "Drawing"], y=[r["Your estimate"], r["Drawing"]],
                             marker_color=[VIOLET, ORANGE], texttemplate="%{y:,.1f}", textposition="outside",
                             cliponaxis=False, showlegend=False), row=1, col=i)
    return style_fig(fig, 340, "Bulk materials: your quantities vs drawing")


def fig_cost_check(est_basis, det_basis, est_cost):
    labels = ["Materials · your quantities", "Materials · drawing quantities", "Your estimated cost"]
    vals = [est_basis, det_basis, est_cost]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=[VIOLET, ORANGE, TEAL],
                           text=[inr(v) for v in vals], textposition="outside", cliponaxis=False))
    fig.update_yaxes(range=[0, max(vals) * 1.25])
    return style_fig(fig, 340, "Is the estimated cost realistic?")


def fig_importance(imp):
    imp = sorted(imp, key=lambda t: t[1])
    fig = go.Figure(go.Bar(x=[v for _, v in imp], y=[n.replace("_", " ") for n, _ in imp], orientation="h",
                           marker_color=VIOLET, texttemplate="%{x:.2f}", textposition="outside", cliponaxis=False))
    return style_fig(fig, 340, "Random-Forest feature importance (not proof of causation)")


def fig_history(df):
    fig = go.Figure()
    x = list(range(1, len(df) + 1))
    colors = [STATUS_META.get(s, (VIOLET,))[0] for s in df["Status"]]
    fig.add_trace(go.Scatter(x=x, y=df["Deviation (%)"], mode="lines+markers", name="Deviation",
                             line=dict(color=VIOLET, width=3), marker=dict(size=11, color=colors,
                                                                          line=dict(color="white", width=2)),
                             customdata=df["Status"], hovertemplate="#%{x} · %{y:.2f}% · %{customdata}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=df["Tolerance (%)"], mode="lines", name="Tolerance",
                             line=dict(color=ROSE, dash="dash")))
    fig.update_xaxes(title="Saved validation #", dtick=1)
    fig.update_yaxes(title="Deviation (%)")
    return style_fig(fig, 340, "Deviation trend across saved validations")


def fig_status_donut(df):
    cnt = df["Status"].value_counts()
    fig = go.Figure(go.Pie(labels=cnt.index, values=cnt.values, hole=.58,
                           marker=dict(colors=[STATUS_META.get(s, (VIOLET,))[0] for s in cnt.index]),
                           textinfo="label+value"))
    return style_fig(fig, 340, "Status distribution").update_layout(showlegend=False)


# ---------------------------------------------------------
# OCR  (cached - runs once per uploaded image)
# ---------------------------------------------------------
ROOM_RE = re.compile(
    r"((?:MASTER|GUEST|KIDS|COMMON)\s+)?(BED\s?ROOM|BEDROOM|KITCHEN|HALL|LIVING(?:\s?ROOM)?|BATH(?:\s?ROOM)?|"
    r"TOILET|DINING(?:\s?ROOM)?|PORCH|VERANDAH|VERANDA|BALCONY|STORE(?:\s?ROOM)?|POOJA(?:\s?ROOM)?|STUDY|"
    r"UTILITY|LOBBY|GARAGE|OFFICE|WASH|DRESS(?:ING)?|LOUNGE|LIVING|BED)(?![A-Z])(?:\s?(\d))?")
PAIR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[Xx×*]\s*(\d+(?:\.\d+)?)")


def _ocr_words(img, config, scale, pad):
    d = pytesseract.image_to_data(img, config=config, output_type=Output.DICT)
    words = []
    for i in range(len(d["text"])):
        t = str(d["text"][i]).strip()
        if not t:
            continue
        try:
            conf = float(d["conf"][i])
        except Exception:
            conf = -1
        if conf < 20:
            continue
        x0 = (d["left"][i] - pad) / scale
        y0 = (d["top"][i] - pad) / scale
        words.append({"t": t, "x0": x0, "y0": y0,
                      "x1": x0 + d["width"][i] / scale, "y1": y0 + d["height"][i] / scale})
    return words


def _group_lines(words):
    lines = []
    for wd in sorted(words, key=lambda k: k["x0"]):
        cy = (wd["y0"] + wd["y1"]) / 2
        hh = max(wd["y1"] - wd["y0"], 1)
        for L in lines:
            lh = max(L["y1"] - L["y0"], 1)
            lcy = (L["y0"] + L["y1"]) / 2
            gap = wd["x0"] - L["x1"]
            if abs(cy - lcy) < 0.6 * max(hh, lh) and gap <= 2.2 * max(hh, lh):   # boxes may overlap
                L["w"].append(wd["t"])
                L["x1"] = max(L["x1"], wd["x1"])
                L["y0"] = min(L["y0"], wd["y0"])
                L["y1"] = max(L["y1"], wd["y1"])
                break
        else:
            lines.append({"w": [wd["t"]], "x0": wd["x0"], "y0": wd["y0"], "x1": wd["x1"], "y1": wd["y1"]})
    for L in lines:
        L["text"] = " ".join(L["w"]).upper()
        L["cx"] = (L["x0"] + L["x1"]) / 2
        L["cy"] = (L["y0"] + L["y1"]) / 2
    return lines


@st.cache_data(show_spinner=False, max_entries=6)
def run_ocr(img_bytes):
    out = {"img_w": 1, "img_h": 1, "tokens": [], "pairs": [], "labels": [], "error": None}
    try:
        arr = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        h, w = arr.shape[:2]
        out["img_w"], out["img_h"] = w, h
        scale = float(min(3.0, max(0.75, 2000.0 / max(h, w))))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # remove long thick wall lines so they do not glue onto the digits
        inv = 255 - th
        klen = max(120, min(inv.shape[0], inv.shape[1]) // 9)   # >> text height, << wall length
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, klen))
        walls = cv2.bitwise_or(cv2.morphologyEx(inv, cv2.MORPH_OPEN, hk), cv2.morphologyEx(inv, cv2.MORPH_OPEN, vk))
        walls = cv2.dilate(walls, np.ones((3, 3), np.uint8), iterations=1)
        clean = 255 - cv2.subtract(inv, walls)
        pad = 30
        clean = cv2.copyMakeBorder(clean, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

        words_a = _ocr_words(clean, "--oem 3 --psm 11", scale, pad)
        words_b = _ocr_words(clean, "--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789.xX", scale, pad)
        diag = math.hypot(w, h)

        # ---- "L x W" pairs ----
        pairs = []
        for L in _group_lines(words_a) + _group_lines(words_b):
            m = PAIR_RE.search(L["text"])
            if not m:
                continue
            l_, w_ = float(m.group(1)), float(m.group(2))
            if any(abs(p["l"] - l_) < 1e-6 and abs(p["w"] - w_) < 1e-6 and
                   math.hypot(p["cx"] - L["cx"], p["cy"] - L["cy"]) < 0.08 * diag for p in pairs):
                continue
            pairs.append({"l": l_, "w": w_, "cx": L["cx"], "cy": L["cy"],
                          "x0": L["x0"], "y0": L["y0"], "x1": L["x1"], "y1": L["y1"]})
        out["pairs"] = pairs

        # ---- room labels ----
        labels = []
        for L in _group_lines(words_a):
            m = ROOM_RE.search(L["text"])
            if m:
                name = re.sub(r"\s+", " ", (m.group(0) or "")).strip().replace("BED ROOM", "BEDROOM")
                labels.append({"name": name, "cx": L["cx"], "cy": L["cy"],
                               "x0": L["x0"], "y0": L["y0"], "x1": L["x1"], "y1": L["y1"]})
        out["labels"] = labels

        # ---- stand-alone numbers (overall dimensions) ----
        tokens = []
        for wd in words_a + words_b:
            t = wd["t"].strip(".,:;")
            if not (re.fullmatch(r"\d+\.\d{1,2}", t) or re.fullmatch(r"\d{3,5}", t)):
                continue
            cx, cy = (wd["x0"] + wd["x1"]) / 2, (wd["y0"] + wd["y1"]) / 2
            if any(p["x0"] - 6 <= cx <= p["x1"] + 6 and p["y0"] - 6 <= cy <= p["y1"] + 6 for p in pairs):
                continue
            v = float(t)
            if any(abs(k["v"] - v) < 1e-6 and math.hypot(k["cx"] - cx, k["cy"] - cy) < 30 for k in tokens):
                continue
            tokens.append({"v": v, "cx": cx, "cy": cy, "x0": wd["x0"], "y0": wd["y0"],
                           "x1": wd["x1"], "y1": wd["y1"]})
        out["tokens"] = tokens
    except Exception as e:  # noqa
        out["error"] = str(e)
    return out


def pick_overall(ocr, uf):
    """Overall plan width/depth from the big numbers written along the outer walls."""
    if not ocr:
        return None, None, None, None
    toks = []
    for t in ocr.get("tokens", []):
        m = t["v"] * uf
        if 0.5 <= m <= 300:
            toks.append({**t, "m": m})
    if not toks:
        return None, None, None, None
    w, h = ocr.get("img_w", 1), ocr.get("img_h", 1)
    horiz = [t for t in toks if t["cy"] < 0.28 * h or t["cy"] > 0.72 * h]
    tw = max(horiz, key=lambda t: t["m"], default=None)
    vert = [t for t in toks if (t["cx"] < 0.28 * w or t["cx"] > 0.72 * w) and t is not tw]
    td = max(vert, key=lambda t: t["m"], default=None)
    rest = sorted([t for t in toks if t is not tw and t is not td], key=lambda t: -t["m"])
    if tw is None and rest:
        tw = rest.pop(0)
    if td is None and rest:
        td = rest.pop(0)
    if tw is None or td is None:
        return None, None, None, None
    return tw["m"], td["m"], tw, td


def rooms_from_ocr(ocr, uf):
    if not ocr:
        return empty_rooms()
    labels = sorted(ocr.get("labels", []), key=lambda r: (round(r["cy"] / 60), r["cx"]))
    pairs = ocr.get("pairs", [])
    diag = math.hypot(ocr.get("img_w", 1), ocr.get("img_h", 1))
    cands = sorted((math.hypot(lb["cx"] - pr["cx"], lb["cy"] - pr["cy"]), i, j)
                   for i, lb in enumerate(labels) for j, pr in enumerate(pairs))
    used_l, used_p, assign = set(), set(), {}
    for dist, i, j in cands:
        if dist <= 0.30 * diag and i not in used_l and j not in used_p:
            used_l.add(i)
            used_p.add(j)
            assign[i] = j
    rows, seen = [], {}

    def uniq(name):
        n = seen.get(name, 0) + 1
        seen[name] = n
        return name.title() if n == 1 else f"{name.title()} {n}"

    def dim(v):
        m = v * uf
        return round(m, 2) if 0.5 <= m <= 200 else 0.0

    for i, lb in enumerate(labels):
        if i in assign:
            p = pairs[assign[i]]
            rows.append([uniq(lb["name"]), dim(p["l"]), dim(p["w"])])
        else:
            rows.append([uniq(lb["name"]), 0.0, 0.0])
    extra = 0
    for j, p in enumerate(pairs):
        if j not in used_p:
            extra += 1
            rows.append([f"Room {extra}", dim(p["l"]), dim(p["w"])])
    return pd.DataFrame(rows, columns=ROOM_COLS) if rows else empty_rooms()


def annotate_drawing(img_bytes, ocr, det):
    arr = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    th = max(2, int(max(arr.shape[:2]) / 450))
    teal, vio, org = (13, 148, 136), (124, 58, 237), (249, 115, 22)
    for lb in ocr.get("labels", []):
        cv2.rectangle(arr, (int(lb["x0"]), int(lb["y0"])), (int(lb["x1"]), int(lb["y1"])), teal, th)
    for p in ocr.get("pairs", []):
        cv2.rectangle(arr, (int(p["x0"]), int(p["y0"])), (int(p["x1"]), int(p["y1"])), vio, th)
    for t in (det.get("tw"), det.get("td")):
        if t:
            cv2.rectangle(arr, (int(t["x0"]) - 4, int(t["y0"]) - 4), (int(t["x1"]) + 4, int(t["y1"]) + 4), org, th + 2)
    return arr


# ---------------------------------------------------------
# SAMPLE DRAWING  (OCR-friendly, generated locally)
# ---------------------------------------------------------
def _font(size, bold=False):
    names = (["arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"] if bold
             else ["arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"])
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def make_sample_plan():
    S = 80  # px per metre
    W, H, ox, oy = 1600, 1350, 220, 170
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = ox, oy, ox + int(15 * S), oy + int(12.5 * S)
    d.rectangle([x0, y0, x1, y1], outline="black", width=16)
    ry = y0 + int(6.0 * S)
    d.line([x0, ry, x1, ry], fill="black", width=9)
    for xm in (7.5, 12.0):
        d.line([x0 + int(xm * S), y0, x0 + int(xm * S), ry], fill="black", width=9)
    for xm in (5.0, 10.0):
        d.line([x0 + int(xm * S), ry, x0 + int(xm * S), y1], fill="black", width=9)

    def center(txt, cx, cy, font):
        b = d.textbbox((0, 0), txt, font=font)
        d.text((cx - (b[2] - b[0]) / 2 - b[0], cy - (b[3] - b[1]) / 2 - b[1]), txt, fill="black", font=font)

    f_big, f_name, f_dim = _font(54, True), _font(34, True), _font(32)
    center("15.00", (x0 + x1) // 2, y0 - 70, f_big)
    center("12.50", ox - 100, (y0 + y1) // 2, f_big)
    rooms = [("LIVING", "7.50 x 6.00", 0, 7.5, 0, 6.0), ("KITCHEN", "4.50 x 6.00", 7.5, 12.0, 0, 6.0),
             ("BATH", "3.00 x 6.00", 12.0, 15.0, 0, 6.0), ("BEDROOM", "5.00 x 6.50", 0, 5.0, 6.0, 12.5),
             ("BEDROOM", "5.00 x 6.50", 5.0, 10.0, 6.0, 12.5), ("DINING", "5.00 x 6.50", 10.0, 15.0, 6.0, 12.5)]
    for name, dm, xa, xb, ya, yb in rooms:
        cx = x0 + int((xa + xb) / 2 * S)
        cy = y0 + int((ya + yb) / 2 * S)
        center(name, cx, cy - 26, f_name)
        center(dm, cx, cy + 26, f_dim)
    d.arc([x0 + 150, ry - 110, x0 + 150 + 110, ry], 90, 180, fill="black", width=5)
    d.arc([x1 - 260, y1 - 110, x1 - 150, y1], 270, 360, fill="black", width=5)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------
# DETECTION  (drawing -> numbers)  &  CALLBACKS
# ---------------------------------------------------------
def clean_rooms(df):
    if df is None or len(df) == 0:
        d = empty_rooms()
    else:
        d = df.copy()
    for c in ("Length (m)", "Width (m)"):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0).clip(lower=0.0)
    d["Room"] = d["Room"].fillna("").astype(str).str.strip()
    d = d[d["Room"] != ""].copy()
    d["Area (m²)"] = d["Length (m)"] * d["Width (m)"]
    d["Perimeter (m)"] = np.where(d["Area (m²)"] > 0, 2 * (d["Length (m)"] + d["Width (m)"]), 0.0)
    return d.reset_index(drop=True)


def get_detected():
    if not ss.img_bytes:
        return None
    ocr = ss.ocr or {}
    uf = UNIT_FACTORS.get(ss.unit, 1.0)
    rooms = clean_rooms(ss.rooms_df)
    W = D = tw = td = None
    source = "manual"
    if ss.calib_mode == "Auto (OCR)":
        W, D, tw, td = pick_overall(ocr, uf)
        if W:
            source = "ocr"
    auto_failed = ss.calib_mode == "Auto (OCR)" and W is None
    if W is None:
        W, D = float(ss.man_w), float(ss.man_d)
    sized = rooms[rooms["Area (m²)"] > 0]
    n = len(rooms)
    outer = 2 * (W + D)
    wall = max((sized["Perimeter (m)"].sum() + outer) / 2, outer) if len(sized) else outer * 1.5
    return {
        "W": W, "D": D, "area": W * D, "wall": float(wall),
        "doors": n + 1 if n else 4, "windows": n if n else 6,
        "columns": (math.ceil(W / COLUMN_SPACING) + 1) * (math.ceil(D / COLUMN_SPACING) + 1),
        "rooms": rooms, "sized": sized, "source": source, "auto_failed": auto_failed,
        "tw": tw, "td": td, "ocr_error": ocr.get("error"),
    }


def sync_estimates():
    det = get_detected()
    if not det:
        return
    ss.est_area = float(max(1.0, round(det["area"], 1)))
    ss.est_wall = float(max(1.0, round(det["wall"], 1)))
    ss.est_doors, ss.est_windows, ss.est_columns = int(det["doors"]), int(det["windows"]), int(det["columns"])


def rebuild_rooms():
    df = rooms_from_ocr(ss.ocr, UNIT_FACTORS.get(ss.unit, 1.0))
    ss.rooms_df = df
    ss.rooms_snapshot = df.copy()
    ss.visit += 1


def on_calibration_change(rebuild=False):
    if rebuild:
        rebuild_rooms()
    sync_estimates()


def load_image(data, name):
    ss.img_bytes, ss.img_name = data, name
    ss.img_sig = hashlib.md5(data).hexdigest()
    with st.spinner("🔎 Reading dimensions & room labels (OCR)…"):
        ss.ocr = run_ocr(data)
    ss.calib_mode = "Auto (OCR)"
    rebuild_rooms()
    sync_estimates()


def clear_image():
    ss.img_bytes = ss.img_name = ss.img_sig = ss.ocr = ss.last_up_sig = None
    ss.rooms_df = empty_rooms()
    ss.rooms_snapshot = empty_rooms()
    ss.up_ver += 1
    ss.visit += 1


# ---------------------------------------------------------
# ML  (cached, optional)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=256)
def ml_predict(feats, dev):
    reg, clf = MODELS["reg"], MODELS["clf"]
    if reg is None or clf is None:
        return None
    try:
        try:
            comp_enc = MODELS["le_comp"].transform(["Floor Area"])[0]
        except Exception:
            comp_enc = 0
        X = pd.DataFrame([dict(zip(FEATURE_COLS, (comp_enc,) + tuple(feats)))])
        Xr = X
        names = getattr(reg, "feature_names_in_", None)
        if names is not None and set(names) <= set(X.columns):
            Xr = X[list(names)]
        Xc = X.copy()
        Xc["Deviation_Percentage"] = dev
        names = getattr(clf, "feature_names_in_", None)
        if names is not None and set(names) <= set(Xc.columns):
            Xc = Xc[list(names)]
        qty = float(reg.predict(Xr)[0])
        enc = clf.predict(Xc)[0]
        status = str(MODELS["le_stat"].inverse_transform([enc])[0])
        proba = clf.predict_proba(Xc)[0]
        try:
            labels = [str(x) for x in MODELS["le_stat"].inverse_transform(clf.classes_)]
        except Exception:
            labels = [str(c) for c in clf.classes_]
        imp = getattr(clf, "feature_importances_", None)
        imp_list = [(n, float(v)) for n, v in zip(Xc.columns, imp)] if imp is not None else []
        return {"qty": qty, "status": status, "conf": float(max(proba) * 100),
                "proba": dict(zip(labels, [float(p) for p in proba])), "imp": imp_list, "error": None}
    except Exception as e:  # noqa
        return {"qty": None, "status": "n/a", "conf": 0.0, "proba": {}, "imp": [], "error": str(e)}


# ---------------------------------------------------------
# MATERIAL TAKE-OFF
# ---------------------------------------------------------
def global_materials(wall_len, area, doors, windows, columns, h, thk, slab, csz):
    net = max(wall_len * h - (doors * DOOR_AREA + windows * WINDOW_AREA), 0.0)
    return {"net_wall_area": net,
            "bricks": math.ceil(net * thk / BRICK_VOL),
            "col_conc": columns * csz * csz * h,
            "slab_conc": area * slab,
            "plaster_bags": math.ceil(net * 2 * PLASTER_T * BAGS_PER_M3)}


ROOM_MAT_COLS = ["Room", "Length (m)", "Width (m)", "Area (m²)", "Perimeter (m)",
                 "Tiles (pcs)", "Wall area (m²)", "Paint (L)", "Cement (bags)"]


def room_materials(sized, h):
    rows = []
    for _, r in sized.iterrows():
        area, per = float(r["Area (m²)"]), float(r["Perimeter (m)"])
        wall = max(per * h - ROOM_OPENINGS, 0.0)
        rows.append([r["Room"], r["Length (m)"], r["Width (m)"], area, per,
                     math.ceil(area / TILE_AREA * TILE_WASTE), wall,
                     wall / PAINT_COVER * PAINT_COATS, area * PLASTER_T * BAGS_PER_M3])
    return pd.DataFrame(rows, columns=ROOM_MAT_COLS)


def compute_analysis():
    det = get_detected()
    if det is None:
        return None
    tol, h = float(ss.tolerance), float(ss.wall_h)
    est = {"area": float(ss.est_area), "wall": float(ss.est_wall), "doors": int(ss.est_doors),
           "windows": int(ss.est_windows), "columns": int(ss.est_columns)}
    comps = [("Floor Area", "m²", det["area"], est["area"]), ("Wall Length", "m", det["wall"], est["wall"]),
             ("Doors", "nos", det["doors"], est["doors"]), ("Windows", "nos", det["windows"], est["windows"]),
             ("Columns", "nos", det["columns"], est["columns"])]
    rows = []
    for name, unit, d_, e_ in comps:
        dv = calc_dev(d_, e_)
        rows.append({"Component": name, "Unit": unit, "Drawing": float(d_), "Your Estimate": float(e_),
                     "Deviation (%)": dv, "Result": comp_result(dv, tol)})
    comp_df = pd.DataFrame(rows)
    overall = float(comp_df["Deviation (%)"].max())
    worst = comp_df.loc[comp_df["Deviation (%)"].idxmax(), "Component"]
    status = "Valid" if overall <= tol else "Minor Difference" if overall <= tol * 1.5 else "Invalid"

    det_qty = det["doors"] + det["windows"] + det["columns"]
    est_qty = est["doors"] + est["windows"] + est["columns"]
    ml = ml_predict((float(det["wall"]), float(det["area"]), float(det_qty), est["wall"], est["area"],
                     float(est_qty), est["area"], tol), overall)

    thk, slab, csz = float(ss.wall_thk), float(ss.slab_thk), float(ss.col_size)
    g_det = global_materials(det["wall"], det["area"], det["doors"], det["windows"], det["columns"], h, thk, slab, csz)
    g_est = global_materials(est["wall"], est["area"], est["doors"], est["windows"], est["columns"], h, thk, slab, csz)
    glob_df = pd.DataFrame([
        ["Bricks (nos)", g_est["bricks"], g_det["bricks"]],
        ["Column concrete (m³)", g_est["col_conc"], g_det["col_conc"]],
        ["Slab concrete (m³)", g_est["slab_conc"], g_det["slab_conc"]],
        ["Wall plaster cement (bags)", g_est["plaster_bags"], g_det["plaster_bags"]],
    ], columns=["Material", "Your estimate", "Drawing"])
    glob_df["Difference"] = glob_df["Your estimate"] - glob_df["Drawing"]
    glob_df["Diff (%)"] = [calc_dev(d_, e_) for d_, e_ in zip(glob_df["Drawing"], glob_df["Your estimate"])]

    rm = room_materials(det["sized"], h)
    unsized = det["rooms"].loc[det["rooms"]["Area (m²)"] <= 0, "Room"].tolist()

    r_ = {k: float(ss[k]) for k in ("rate_brick", "rate_cement", "rate_concrete", "rate_tile", "rate_paint")}

    def gcost(g):
        return {"Bricks": g["bricks"] * r_["rate_brick"], "Column concrete": g["col_conc"] * r_["rate_concrete"],
                "Slab concrete": g["slab_conc"] * r_["rate_concrete"],
                "Wall plaster cement": g["plaster_bags"] * r_["rate_cement"]}

    room_cost = {"Floor tiles": float(rm["Tiles (pcs)"].sum()) * r_["rate_tile"],
                 "Wall paint": float(rm["Paint (L)"].sum()) * r_["rate_paint"],
                 "Ceiling plaster cement": math.ceil(float(rm["Cement (bags)"].sum())) * r_["rate_cement"]}
    cost_det_items = {**gcost(g_det), **room_cost}
    cost_det, cost_est = sum(cost_det_items.values()), sum({**gcost(g_est), **room_cost}.values())

    A = {"det": det, "est": est, "tol": tol, "h": h, "comp_df": comp_df, "overall_dev": overall,
         "worst": worst, "status": status, "ml": ml, "g_det": g_det, "g_est": g_est, "glob_df": glob_df,
         "room_mat": rm, "unsized": unsized, "cost_det_items": cost_det_items, "cost_det": cost_det,
         "cost_est_basis": cost_est, "est_cost": float(ss.est_cost)}
    A["recs"] = build_recs(A)
    return A


def build_recs(A):
    tol, det, est, cdf = A["tol"], A["det"], A["est"], A["comp_df"]
    recs = []
    if A["status"] == "Valid":
        recs.append(("ok", "Estimation within tolerance",
                     f"All five components are within ±{tol:.0f}% of the drawing values."))
    for _, r in cdf.iterrows():
        if r["Deviation (%)"] > tol:
            over = r["Your Estimate"] > r["Drawing"]
            sev = "high" if r["Deviation (%)"] > tol * 1.5 else "warn"
            recs.append((sev, f'{r["Component"]}: {"overestimated" if over else "underestimated"}',
                         f'You entered {r["Your Estimate"]:,.1f} {r["Unit"]}; the drawing suggests ≈ '
                         f'{r["Drawing"]:,.1f} {r["Unit"]} (deviation {r["Deviation (%)"]:.1f}% vs {tol:.0f}% allowed).'))
    g_d, g_e = A["g_det"], A["g_est"]
    if calc_dev(det["wall"], est["wall"]) > tol:
        recs.append(("info", "Brick impact",
                     f'Your wall length gives ≈ {g_e["bricks"]:,} bricks, the drawing gives ≈ {g_d["bricks"]:,} '
                     f'({g_e["bricks"] - g_d["bricks"]:+,}).'))
    if calc_dev(det["columns"], est["columns"]) > tol:
        recs.append(("info", "Column concrete impact",
                     f'{g_e["col_conc"]:.2f} m³ (yours) vs {g_d["col_conc"]:.2f} m³ (drawing).'))
    if calc_dev(det["area"], est["area"]) > tol:
        recs.append(("info", "Slab concrete impact",
                     f'{g_e["slab_conc"]:.2f} m³ (yours) vs {g_d["slab_conc"]:.2f} m³ (drawing).'))
    if A["unsized"]:
        recs.append(("warn", "Rooms without sizes",
                     "Not included in room-wise materials: " + ", ".join(A["unsized"]) + "."))
    if A["est_cost"] < A["cost_det"]:
        recs.append(("high", "Estimated cost looks too low",
                     f'Your estimated cost {inr(A["est_cost"])} is below the drawing-based material cost alone '
                     f'({inr(A["cost_det"])}, using the editable placeholder rates). Labour and overheads come on top.'))
    elif A["status"] != "Valid":
        recs.append(("info", "Cost impact",
                     f'Because quantities deviate from the drawing, {inr(A["est_cost"])} may be inaccurate.'))
    return recs


def risk_summary(A):
    lvl, col = {"Valid": ("Low", EMERALD), "Minor Difference": ("Medium", AMBER), "Invalid": ("High", ROSE)}[A["status"]]
    if A["status"] == "Valid":
        reason, nxt = "All component deviations are inside the tolerance.", "Save the result and export the report."
    else:
        r = A["comp_df"].sort_values("Deviation (%)", ascending=False).iloc[0]
        reason = f'{r["Component"]} deviates by {r["Deviation (%)"]:.1f}% ({"over" if r["Your Estimate"] > r["Drawing"] else "under"}-estimated).'
        nxt = f'Re-check {r["Component"].lower()}; fix the rooms table if OCR misread a size.'
    return lvl, col, reason, nxt


def save_history(A):
    sig = (round(A["est"]["area"], 2), round(A["det"]["area"], 2), A["overall_dev"], A["tol"], ss.img_sig,
           round(A["est"]["wall"], 2), A["est"]["doors"], A["est"]["windows"], A["est"]["columns"])
    if ss.last_saved_sig == sig:
        return False
    ss.last_saved_sig = sig
    ss.history.append({
        "Date": time.strftime("%Y-%m-%d"), "Time": time.strftime("%H:%M:%S"),
        "Drawing": ss.img_name, "Est. Area (m²)": round(A["est"]["area"], 2),
        "Det. Area (m²)": round(A["det"]["area"], 2), "Deviation (%)": A["overall_dev"],
        "Tolerance (%)": A["tol"], "Status": A["status"],
        "ML Status": A["ml"]["status"] if A["ml"] and not A["ml"]["error"] else "n/a",
        "Rooms": len(A["det"]["rooms"]), "Material Cost (₹)": round(A["cost_det"]),
    })
    return True


# ---------------------------------------------------------
# PDF REPORT  (robust: fpdf & fpdf2, no temp-file leaks, no nested buttons)
# ---------------------------------------------------------
def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return str(o)


def build_payload(A):
    return {
        "generated": time.strftime("%d %b %Y, %H:%M"), "drawing": str(ss.img_name or "Drawing"),
        "status": A["status"], "overall": A["overall_dev"], "tol": A["tol"],
        "plan": f'{A["det"]["W"]:.2f} x {A["det"]["D"]:.2f} m', "det_area": A["det"]["area"],
        "est_area": A["est"]["area"], "est_cost": A["est_cost"], "cost_det": A["cost_det"],
        "cost_est_basis": A["cost_est_basis"], "wall_h": A["h"],
        "ml": ({"status": A["ml"]["status"], "conf": A["ml"]["conf"], "qty": A["ml"]["qty"]}
               if A["ml"] and not A["ml"]["error"] else None),
        "components": A["comp_df"].to_dict("records"), "rooms": A["room_mat"].to_dict("records"),
        "glob": A["glob_df"].to_dict("records"), "recs": [{"sev": s, "title": t, "text": x} for s, t, x in A["recs"]],
    }


def _pdf_safe(s):
    s = str(s).replace("**", "")
    for a, b in {"₹": "Rs. ", "≈": "~", "≤": "<=", "≥": ">=", "→": "->", "–": "-", "—": "-", "’": "'",
                 "“": '"', "”": '"', "✅": "[OK]", "⚠️": "[!]", "❌": "[X]", "ℹ️": "[i]", "³": "3"}.items():
        s = s.replace(a, b)
    return s.encode("latin-1", "ignore").decode("latin-1")



try:  # fpdf2 (modern API)
    from fpdf.enums import XPos, YPos
    _NEW_FPDF = True
except Exception:  # legacy fpdf
    _NEW_FPDF = False


def _cell(pdf, w, h, txt, border=0, ln=0, align="", fill=False):
    """cell() that works on both fpdf (old) and fpdf2 without deprecation noise."""
    if _NEW_FPDF:
        if ln == 1:
            pdf.cell(w, h, txt, border, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align=align, fill=fill)
        else:
            pdf.cell(w, h, txt, border, new_x=XPos.RIGHT, new_y=YPos.TOP, align=align, fill=fill)
    else:
        pdf.cell(w, h, txt, border, ln, align, fill)


class PDFReport(FPDF):
    def header(self):
        self.set_fill_color(109, 40, 217)
        self.rect(0, 0, 210, 16, "F")
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 4)
        _cell(self, 190, 8, "Engineering Drawing Validation Report", 0, 0, "L")
        self.set_text_color(43, 33, 64)
        self.set_y(22)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 110, 140)
        _cell(self, 0, 8, f"Page {self.page_no()}", 0, 0, "C")


def _h(pdf, text):
    pdf.set_x(pdf.l_margin)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(109, 40, 217)
    _cell(pdf, 0, 8, _pdf_safe(text), 0, 1, "L")
    pdf.set_text_color(43, 33, 64)
    pdf.set_font("Helvetica", "", 10)


def _mc(pdf, text, h=6):
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, h, _pdf_safe(text))
    pdf.set_x(pdf.l_margin)


def _tbl(pdf, headers, rows, widths):
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(237, 233, 254)
    pdf.set_text_color(60, 30, 120)
    for w, hd in zip(widths, headers):
        _cell(pdf, w, 7, _pdf_safe(hd), 1, 0, "C", True)
    pdf.ln(7)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(43, 33, 64)
    for i, row in enumerate(rows):
        pdf.set_fill_color(*((250, 245, 255) if i % 2 == 0 else (255, 255, 255)))
        pdf.set_x(pdf.l_margin)
        for w, v in zip(widths, row):
            _cell(pdf, w, 6.5, _pdf_safe(v)[: int(w / 1.9)], 1, 0, "C", True)
        pdf.ln(6.5)


def _png_path(fig):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    f.close()
    fig.savefig(f.name, dpi=140, bbox_inches="tight")
    return f.name


def _pdf_charts(P):
    paths = []
    comps = P["components"]
    tol = P["tol"]
    fig = Figure(figsize=(10, 3.8))
    ax1, ax2 = fig.subplots(1, 2)
    x = np.arange(2)
    ax1.bar(x - .18, [comps[0]["Your Estimate"], comps[1]["Your Estimate"]], .36, color=VIOLET, label="Your estimate")
    ax1.bar(x + .18, [comps[0]["Drawing"], comps[1]["Drawing"]], .36, color=ORANGE, label="Drawing")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Area (m²)", "Wall length (m)"])
    ax1.legend()
    ax1.set_title("Estimate vs drawing")
    devs = [c["Deviation (%)"] for c in comps]
    cols = [EMERALD if d <= tol else AMBER if d <= tol * 1.5 else ROSE for d in devs]
    ax2.bar([c["Component"] for c in comps], devs, color=cols)
    ax2.axhline(tol, color=ROSE, linestyle="--", label=f"Tolerance {tol:.0f}%")
    for i, d in enumerate(devs):
        ax2.text(i, d, f"{d:.1f}%", ha="center", va="bottom", fontsize=8)
    ax2.legend()
    ax2.set_title("Deviation per component (%)")
    fig.tight_layout()
    paths.append(_png_path(fig))
    rooms = P["rooms"]
    if rooms:
        fig2 = Figure(figsize=(10, 3.4))
        a1, a2 = fig2.subplots(1, 2)
        names = [r["Room"] for r in rooms]
        cc = [COLORWAY[i % len(COLORWAY)] for i in range(len(names))]
        a1.barh(names, [r["Area (m²)"] for r in rooms], color=cc)
        a1.set_title("Room area (m²)")
        a2.bar(names, [r["Tiles (pcs)"] for r in rooms], color=cc)
        a2.set_title("Floor tiles per room (pcs)")
        a2.tick_params(axis="x", labelrotation=30, labelsize=8)
        fig2.tight_layout()
        paths.append(_png_path(fig2))
    return paths


def _place_image(pdf, path, w=180):
    try:
        iw, ih = Image.open(path).size
        h = w * ih / iw
        if pdf.get_y() + h > pdf.page_break_trigger:
            pdf.add_page()
        pdf.image(path, x=(210 - w) / 2, y=pdf.get_y(), w=w)
        pdf.set_y(pdf.get_y() + h + 4)
    except Exception:
        pass


@st.cache_data(show_spinner=False, max_entries=8)
def build_pdf(payload_json, img_bytes):
    P = json.loads(payload_json)
    tmp = []
    pdf = PDFReport("P", "mm", "A4")
    pdf.set_auto_page_break(True, margin=16)
    pdf.add_page()
    pdf.set_font("Helvetica", "", 10)
    _mc(pdf, f'Drawing: {P["drawing"]}    Generated: {P["generated"]}')

    _h(pdf, "1. Validation Summary")
    _tbl(pdf, ["Item", "Value"], [
        ["Validation status", P["status"]],
        ["Overall deviation", f'{P["overall"]:.2f}% (tolerance {P["tol"]:.0f}%)'],
        ["Plan size (W x D)", P["plan"]],
        ["Detected floor area", f'{P["det_area"]:.2f} m2'],
        ["Estimated floor area", f'{P["est_area"]:.2f} m2'],
        ["Estimated cost", inr(P["est_cost"])],
        ["Material cost (drawing qty)", inr(P["cost_det"])],
        ["ML classifier", f'{P["ml"]["status"]} ({P["ml"]["conf"]:.1f}% confidence)' if P["ml"] else "not available"],
    ], [70, 120])

    _h(pdf, "2. Component Comparison")
    rows = [[c["Component"], c["Unit"], f'{c["Drawing"]:.2f}', f'{c["Your Estimate"]:.2f}',
             f'{c["Deviation (%)"]:.2f}', c["Result"]] for c in P["components"]]
    _tbl(pdf, ["Component", "Unit", "Drawing", "Estimate", "Dev %", "Result"], rows, [32, 16, 30, 30, 24, 58])

    _h(pdf, "3. Recommendations")
    for r in P["recs"]:
        pdf.set_font("Helvetica", "B", 10)
        _mc(pdf, {"ok": "[OK] ", "warn": "[!] ", "high": "[X] ", "info": "[i] "}.get(r["sev"], "") + r["title"])
        pdf.set_font("Helvetica", "", 10)
        _mc(pdf, "    " + r["text"])

    _h(pdf, "4. Bulk Materials")
    _tbl(pdf, ["Material", "Your estimate", "Drawing", "Difference", "Diff %"],
         [[g["Material"], f'{g["Your estimate"]:,.2f}', f'{g["Drawing"]:,.2f}', f'{g["Difference"]:+,.2f}',
           f'{g["Diff (%)"]:.1f}'] for g in P["glob"]], [62, 32, 32, 32, 32])

    if P["rooms"]:
        _h(pdf, "5. Room-wise Materials")
        _tbl(pdf, ["Room", "L x W (m)", "Area m2", "Tiles", "Paint L", "Cement bags"],
             [[r["Room"], f'{r["Length (m)"]:.2f} x {r["Width (m)"]:.2f}', f'{r["Area (m²)"]:.2f}',
               f'{int(r["Tiles (pcs)"])}', f'{r["Paint (L)"]:.1f}', f'{r["Cement (bags)"]:.1f}'] for r in P["rooms"]],
             [40, 36, 28, 24, 28, 34])

    _h(pdf, "6. Charts")
    try:
        tmp = _pdf_charts(P)
        for p in tmp:
            _place_image(pdf, p)
    except Exception:
        _mc(pdf, "(charts could not be rendered)")

    if img_bytes:
        try:
            f = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            f.close()
            Image.open(io.BytesIO(img_bytes)).convert("RGB").save(f.name, "JPEG", quality=85)
            tmp.append(f.name)
            pdf.add_page()
            _h(pdf, "7. Drawing")
            _place_image(pdf, f.name, 170)
        except Exception:
            pass

    out = pdf.output() if _NEW_FPDF else pdf.output(dest="S")
    data = out.encode("latin-1") if isinstance(out, str) else bytes(out)
    for p in tmp:
        try:
            os.remove(p)
        except Exception:
            pass
    return data


# ---------------------------------------------------------
# PAGE: DASHBOARD
# ---------------------------------------------------------
def render_dashboard(A):
    hero("Engineering Drawing Validation",
         "Cross-check your quantity estimates against the floor plan - OCR, machine learning and material take-off in one live dashboard.",
         "Dashboard")

    has_img, has_rooms = bool(ss.img_bytes), bool(A and len(A["det"]["rooms"]))
    flags = [has_img, A is not None, has_rooms, A is not None, len(ss.history) > 0]
    labels = [("Upload", "Drawing image"), ("Calibrate", "Plan dimensions"), ("Rooms", "Detected & editable"),
              ("Validate", "Live deviation check"), ("Save", "History & report")]
    cur = next((i for i, f in enumerate(flags) if not f), len(flags))
    steps = []
    for i, ((t, s), f) in enumerate(zip(labels, flags)):
        cls = "done" if f else ("now" if i == cur else "")
        steps.append(f'<div class="step {cls}" style="animation-delay:{i * 80}ms"><div class="step-n">{"✓" if f else i + 1}</div>'
                     f'<div><div class="step-t">{t}</div><div class="step-s">{s}</div></div></div>')
    html('<div class="steps">' + "".join(steps) + "</div>")

    dash = "—"
    items = [
        kpi("🖼️", "Drawing", esc(str(ss.img_name)) if has_img else dash,
            "OCR source: " + ("auto" if A and A["det"]["source"] == "ocr" else "manual") if A else "Upload on the Validate page", VIOLET, 0),
        kpi("📐", "Plan area", f'{A["det"]["area"]:.1f} m²' if A else dash,
            f'{A["det"]["W"]:.2f} × {A["det"]["D"]:.2f} m' if A else "waiting for drawing", ORANGE, 70),
        kpi("🏠", "Rooms detected", str(len(A["det"]["rooms"])) if A else dash,
            f'{len(A["det"]["sized"])} with sizes' if A else "waiting for drawing", TEAL, 140),
        kpi("🎯", "Overall deviation", f'{A["overall_dev"]:.1f}%' if A else dash,
            f'{A["status"]} · tolerance {A["tol"]:.0f}%' if A else "no validation yet",
            STATUS_META[A["status"]][0] if A else AMBER, 210),
        kpi("🧱", "Material cost", inr(A["cost_det"]) if A else dash, "drawing basis · placeholder rates", PINK, 280),
        kpi("📜", "Saved validations", str(len(ss.history)), "in this session", EMERALD, 350),
    ]
    kpi_grid(items)

    if A is None:
        empty_state("📐", "Ready for your first drawing",
                    "Upload an OCR-friendly floor plan (or use the built-in sample) and everything on this dashboard goes live.")
        st.write("")
        c1, c2, c3 = st.columns([1, 1.2, 1])
        c2.button("🚀 Start Validation", on_click=goto, args=(PAGE_VAL,), key="dash_start")
    else:
        status_banner(A)
        c1, c2 = st.columns([1, 1.5], gap="large")
        with c1:
            plot(fig_gauge(A["overall_dev"], A["tol"]), "dash_gauge")
        with c2:
            plot(fig_compare(A["comp_df"]), "dash_compare")
        c3, c4 = st.columns(2, gap="large")
        with c3:
            if len(A["room_mat"]):
                plot(fig_room_share(A["room_mat"]), "dash_share")
            else:
                callout("info", "Room chart waiting", "Room sizes appear here once detected or entered.")
        with c4:
            plot(fig_cost_donut(A["cost_det_items"]), "dash_cost")

    if ss.history:
        section("📈", "Validation trend", "Your saved validations in this session")
        plot(fig_history(pd.DataFrame(ss.history)), "dash_hist")

    section("⚡", "Quick actions")
    q1, q2, q3, q4 = st.columns(4)
    q1.button("📐 Validate Drawing", on_click=goto, args=(PAGE_VAL,), key="qa1")
    q2.button("🧱 Materials", on_click=goto, args=(PAGE_MAT,), key="qa2")
    q3.button("📥 Export Report", on_click=goto, args=(PAGE_REP,), key="qa3")
    q4.button("📜 History", on_click=goto, args=(PAGE_HIS,), key="qa4")
    footer()


# ---------------------------------------------------------
# PAGE: VALIDATE
# ---------------------------------------------------------
GUIDE_MD = """
**Make the drawing easy for OCR**

- **Black lines on a pure white background**, flat 2D top view. No colours, shading, grid, furniture, 3D or title block.
- **Big, bold, horizontal text** (never rotated), sans-serif, at least ~3 % of the image height.
- **Overall dimensions:** write two big numbers with 2 decimals, e.g. `15.00` *above/below* the plan (width) and `12.50` *left/right* of the plan (depth), horizontally.
- **Rooms:** UPPERCASE name + size, e.g. `KITCHEN` and under it `3.00 x 3.50` (use a plain letter **x**).
- Units: metres by default (or mm with 4-digit numbers like `3000`, or feet – pick the unit in *Scale Calibration*).
- Keep ≥ 2000 px on the long side, 5–8 rooms, wide white margin; no dimension lines/arrows crossing the digits.
- AI image generators often garble digits – **always check the numbers**, then fix anything wrong in the *Detected Rooms* table.
"""


def render_validate():
    hero("Drawing Estimation Validation",
         "Upload a floor plan, calibrate the scale and enter your estimates - every graph and material updates live.",
         "Validate Drawing")
    if MODELS["error"]:
        callout("warn", "ML model files not loaded",
                "Validation still works with the tolerance rules. Run model.py to enable the Random-Forest prediction.")

    colL, colR = st.columns([1.05, 1], gap="large")

    # ------------------------------ LEFT: upload + calibrate
    with colL:
        section("📤", "1 · Upload & Calibrate", "OCR reads overall dimensions and room labels automatically")
        with st.expander("📸 How should the drawing look for best OCR?"):
            st.markdown(GUIDE_MD)

        up = st.file_uploader("Upload civil drawing (JPG / PNG)", type=["jpg", "jpeg", "png"],
                              key=f"uploader_{ss.up_ver}")
        if up is not None:
            data = up.getvalue()
            sig = hashlib.md5(data).hexdigest()
            if sig != ss.last_up_sig:
                ss.last_up_sig = sig
                load_image(data, up.name)

        b1, b2 = st.columns(2)
        if b1.button("🧪 Use sample drawing", key="btn_sample"):
            load_image(make_sample_plan(), "sample_floorplan.png")
        if ss.img_bytes and b2.button("🗑️ Remove drawing", key="btn_remove"):
            clear_image()
            st.rerun()

        if ss.img_bytes:
            img = Image.open(io.BytesIO(ss.img_bytes)).convert("RGB")
            det0 = get_detected()
            t1, t2 = st.tabs(["📄 Original drawing", "🎯 Detected & marked"])
            with t1:
                show_image(img, f"{ss.img_name}")
            with t2:
                if ss.ocr and not ss.ocr.get("error"):
                    show_image(annotate_drawing(ss.img_bytes, ss.ocr, det0), "OCR detections")
                    html('<div class="legend"><span><i style="background:#0D9488"></i>Room labels</span>'
                         '<span><i style="background:#7C3AED"></i>Room sizes</span>'
                         '<span><i style="background:#F97316"></i>Overall dimensions</span></div>')
                else:
                    callout("warn", "OCR is not available",
                            "Install Tesseract (or fix its path) and use Manual Entry meanwhile.")

            section("🎯", "Scale Calibration", "Defines the drawing's true size - changing it re-syncs your estimates")
            st.radio("Dimension source", ["Auto (OCR)", "Manual Entry"], key="calib_mode", horizontal=True,
                     on_change=on_calibration_change, args=(False,))
            st.selectbox("Drawing units", list(UNIT_FACTORS), key="unit",
                         on_change=on_calibration_change, args=(True,))
            det0 = get_detected()
            if ss.calib_mode == "Manual Entry" or det0["auto_failed"]:
                if det0["auto_failed"]:
                    callout("warn", "Overall dimensions not read",
                            "OCR could not find the two big outer numbers. Enter the plan size below (metres).")
                m1, m2 = st.columns(2)
                m1.number_input("Overall width (m)", min_value=0.5, max_value=500.0, step=0.5, key="man_w",
                                on_change=on_calibration_change, args=(False,))
                m2.number_input("Overall depth (m)", min_value=0.5, max_value=500.0, step=0.5, key="man_d",
                                on_change=on_calibration_change, args=(False,))
            else:
                callout("ok", f'Plan size read by OCR: {det0["W"]:.2f} × {det0["D"]:.2f} m',
                        f'Footprint {det0["area"]:.1f} m² · image {ss.ocr["img_w"]}×{ss.ocr["img_h"]} px '
                        f'(≈ {det0["W"] / max(ss.ocr["img_w"], 1) * 1000:.1f} mm per pixel across the width).')
        else:
            empty_state("🖼️", "No drawing yet", "Upload a floor plan or click ‘Use sample drawing’ to try the full flow.")

    # ------------------------------ RIGHT: estimations (live)
    with colR:
        section("🧮", "2 · Your Estimations", "Change anything - results update instantly, no submit button")
        st.markdown("**📏 Quantities**")
        q1, q2, q3 = st.columns(3)
        q1.number_input("Floor area (m²)", min_value=1.0, step=1.0, key="est_area")
        q2.number_input("Wall length (m)", min_value=1.0, step=1.0, key="est_wall")
        q3.number_input("Wall height (m)", min_value=1.0, step=0.5, key="wall_h")
        st.markdown("**🚪 Openings & structure**")
        o1, o2, o3 = st.columns(3)
        o1.number_input("Doors", min_value=0, step=1, key="est_doors")
        o2.number_input("Windows", min_value=0, step=1, key="est_windows")
        o3.number_input("Columns", min_value=0, step=1, key="est_columns")
        st.markdown("**💰 Cost & tolerance**")
        st.number_input("Estimated cost (₹)", min_value=1000, step=1000, key="est_cost")
        st.slider("Acceptable tolerance (%)", min_value=1, max_value=30, key="tolerance")
        with st.expander("⚙️ Construction assumptions (drive the materials)"):
            a1, a2, a3 = st.columns(3)
            a1.number_input("Wall thickness (m)", min_value=0.1, max_value=0.5, step=0.005, format="%.3f", key="wall_thk")
            a2.number_input("Slab thickness (m)", min_value=0.1, max_value=0.4, step=0.005, format="%.3f", key="slab_thk")
            a3.number_input("Column size (m)", min_value=0.15, max_value=0.9, step=0.05, format="%.2f", key="col_size")
        if ss.img_bytes:
            st.button("↺ Sync estimates with drawing", on_click=sync_estimates, key="btn_sync")
        else:
            callout("info", "Waiting for a drawing", "Your inputs are saved. Upload a drawing to start comparing.")

    if not ss.img_bytes:
        footer()
        return None

    # ------------------------------ rooms editor
    section("🏠", "3 · Detected Rooms (editable)",
            "OCR result - correct a name or size and all graphs & materials below recalculate")
    cfg = {"Room": st.column_config.TextColumn("Room", required=True),
           "Length (m)": st.column_config.NumberColumn("Length (m)", min_value=0.0, step=0.1, format="%.2f"),
           "Width (m)": st.column_config.NumberColumn("Width (m)", min_value=0.0, step=0.1, format="%.2f")}
    edited = editor(ss.rooms_snapshot, f"rooms_editor_{ss.visit}", cfg)
    ss.rooms_df = edited

    A = compute_analysis()

    # ------------------------------ results
    section("🎯", "4 · Validation Results", "Rule-based tolerance check + Random-Forest comparison")
    status_banner(A)
    kpi_grid([
        kpi("📐", "Detected area", f'{A["det"]["area"]:.2f} m²', f'{A["det"]["W"]:.2f} × {A["det"]["D"]:.2f} m', ORANGE),
        kpi("🧮", "Estimated area", f'{A["est"]["area"]:.2f} m²', "your input", VIOLET, 60),
        kpi("🎯", "Overall deviation", f'{A["overall_dev"]:.2f}%', f'{A["tol"]:.0f}% allowed', STATUS_META[A["status"]][0], 120),
        kpi(STATUS_META[A["status"]][2], "Status", A["status"], f'worst: {esc(A["worst"])}', STATUS_META[A["status"]][0], 180),
    ])
    g1, g2 = st.columns([1, 1.5], gap="large")
    with g1:
        plot(fig_gauge(A["overall_dev"], A["tol"]), "val_gauge")
    with g2:
        plot(fig_compare(A["comp_df"]), "val_compare")
    plot(fig_dev(A["comp_df"], A["tol"]), "val_dev")

    disp = A["comp_df"].copy()
    for c in ("Drawing", "Your Estimate"):
        disp[c] = [f"{v:.2f}" if u != "nos" else f"{int(round(v))}" for v, u in zip(disp[c], disp["Unit"])]
    table(disp)

    section("🤖", "Algorithm comparison", "Two independent checks of the same inputs")
    ml = A["ml"]
    if ml and not ml["error"]:
        table(pd.DataFrame([["Rule-based tolerance check", A["status"], "deterministic"],
                            ["Random-Forest classifier", ml["status"], f'{ml["conf"]:.1f}%']],
                           columns=["Model", "Prediction", "Confidence"]))
        if ml["status"] == A["status"]:
            callout("ok", "Both checks agree", f'Predicted expected quantity from the regressor: {ml["qty"]:.1f}.')
        else:
            callout("warn", "The checks disagree",
                    f'Rule says {A["status"]}, ML says {ml["status"]}. The tolerance rule is the authoritative result; '
                    f'regressor expected quantity: {ml["qty"]:.1f}.')
        if ml["imp"]:
            with st.expander("🔍 Why this prediction?"):
                plot(fig_importance(ml["imp"]), "val_imp")
    elif ml and ml["error"]:
        callout("warn", "ML prediction unavailable", "The model files did not match the expected inputs. Rule-based result is shown.")
    else:
        callout("info", "ML models not loaded", "Only the rule-based check is shown.")

    section("🏠", "5 · Room-wise Materials", "Automatic - one card per detected room")
    room_cards(A["room_mat"], A["unsized"])
    if len(A["room_mat"]):
        r1, r2 = st.columns([1, 1.7], gap="large")
        with r1:
            plot(fig_room_share(A["room_mat"]), "val_share")
        with r2:
            plot(fig_room_materials(A["room_mat"]), "val_roommat")

    section("💡", "6 · Recommendation", "Derived from your actual inputs")
    lvl, col, reason, nxt = risk_summary(A)
    html(f'<div class="risk"><div style="--c:{col}"><small>⚠️ Risk level</small><b>{lvl}</b></div>'
         f'<div style="--c:{VIOLET}"><small>📌 Reason</small><b>{esc(reason)}</b></div>'
         f'<div style="--c:{TEAL}"><small>✅ Suggested next step</small><b>{esc(nxt)}</b></div></div>')
    for sev, t, x in A["recs"]:
        callout(sev, t, x)

    st.write("")
    s1, s2, s3 = st.columns(3)
    if s1.button("💾 Save to History", key="btn_save"):
        if save_history(A):
            st.toast("Saved to history ✅")
        else:
            st.toast("Nothing new to save - identical to the last saved result.")
    s2.button("🧱 Open Materials", on_click=goto, args=(PAGE_MAT,), key="val_to_mat")
    s3.button("📥 Export Report", on_click=goto, args=(PAGE_REP,), key="val_to_rep")
    footer()
    return A


# ---------------------------------------------------------
# PAGE: MATERIALS
# ---------------------------------------------------------
def render_materials(A):
    hero("Material Estimation", "Bulk and room-wise quantities, cost split and a reality check of your estimated cost.",
         "Materials")
    if A is None:
        empty_state("🧱", "No drawing loaded", "Load a drawing on the Validate page to see the material take-off here.")
        st.write("")
        c1, c2, c3 = st.columns([1, 1.2, 1])
        c2.button("📐 Go to Validate Drawing", on_click=goto, args=(PAGE_VAL,), key="mat_go")
        footer()
        return
    rm, gd = A["room_mat"], A["g_det"]
    kpi_grid([
        kpi("🧱", "Bricks (drawing)", f'{gd["bricks"]:,}', f'yours: {A["g_est"]["bricks"]:,}', ORANGE),
        kpi("🏛️", "Column concrete", f'{gd["col_conc"]:.2f} m³', f'{int(A["det"]["columns"])} columns', VIOLET, 60),
        kpi("🏗️", "Slab concrete", f'{gd["slab_conc"]:.2f} m³', f'{ss.slab_thk * 1000:.0f} mm slab', TEAL, 120),
        kpi("🔲", "Floor tiles", f'{int(rm["Tiles (pcs)"].sum()):,} pcs' if len(rm) else "—", "600×600 mm + 5% wastage", PINK, 180),
        kpi("🎨", "Wall paint", f'{rm["Paint (L)"].sum():.1f} L' if len(rm) else "—", f"{PAINT_COATS} coats", AMBER, 240),
        kpi("💰", "Material cost", inr(A["cost_det"]), "drawing basis", EMERALD, 300),
    ])

    section("🏗️", "Bulk materials", "Your quantities vs quantities implied by the drawing")
    plot(fig_global(A["glob_df"]), "mat_global")
    g = A["glob_df"].copy()
    for c in ("Your estimate", "Drawing", "Difference"):
        g[c] = g[c].map(lambda v: f"{v:,.2f}")
    table(g)

    section("🏠", "Room-wise materials", "Updates automatically when rooms, wall height or assumptions change")
    room_cards(rm, A["unsized"])
    if len(rm):
        c1, c2 = st.columns([1, 1.7], gap="large")
        with c1:
            plot(fig_room_share(rm), "mat_share")
        with c2:
            plot(fig_room_materials(rm), "mat_roommat")
        t = rm.copy()
        for c in ("Length (m)", "Width (m)", "Area (m²)", "Perimeter (m)", "Wall area (m²)", "Paint (L)", "Cement (bags)"):
            t[c] = t[c].map(lambda v: f"{v:.2f}")
        table(t)

    section("💰", "Cost split & reality check", "Uses editable placeholder rates - replace with your local schedule of rates")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        plot(fig_cost_donut(A["cost_det_items"]), "mat_donut")
    with c2:
        plot(fig_cost_check(A["cost_est_basis"], A["cost_det"], A["est_cost"]), "mat_costcheck")
    if A["est_cost"] < A["cost_det"]:
        callout("high", "Estimated cost is below the materials alone",
                f'{inr(A["est_cost"])} < {inr(A["cost_det"])}. Labour, equipment and overheads are not included here.')
    else:
        callout("ok", "Estimated cost covers the drawing-based materials",
                f'Materials are {A["cost_det"] / A["est_cost"] * 100:.0f}% of your estimated cost; the rest must cover labour, '
                f'finishes and overheads.')
    with st.expander("🏷️ Edit material rates (₹) - placeholders, not market prices"):
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.number_input("Brick (₹/no)", min_value=0.0, step=0.5, key="rate_brick")
        r2.number_input("Cement (₹/bag)", min_value=0.0, step=10.0, key="rate_cement")
        r3.number_input("Concrete (₹/m³)", min_value=0.0, step=100.0, key="rate_concrete")
        r4.number_input("Tile (₹/pc)", min_value=0.0, step=5.0, key="rate_tile")
        r5.number_input("Paint (₹/L)", min_value=0.0, step=10.0, key="rate_paint")
    footer()


# ---------------------------------------------------------
# PAGE: REPORT
# ---------------------------------------------------------
def render_report(A):
    hero("Export Report", "A complete PDF (tables, charts, recommendations, drawing) plus a CSV summary - ready to download.",
         "Export Report")
    if A is None:
        empty_state("📥", "Nothing to export yet", "Load a drawing and run a validation first - the report is built from it.")
        st.write("")
        c1, c2, c3 = st.columns([1, 1.2, 1])
        c2.button("📐 Go to Validate Drawing", on_click=goto, args=(PAGE_VAL,), key="rep_go")
        footer()
        return
    status_banner(A)
    kpi_grid([
        kpi("📐", "Detected area", f'{A["det"]["area"]:.2f} m²', "", ORANGE),
        kpi("🧮", "Estimated area", f'{A["est"]["area"]:.2f} m²', "", VIOLET, 60),
        kpi("🎯", "Deviation", f'{A["overall_dev"]:.2f}%', f'{A["tol"]:.0f}% allowed', STATUS_META[A["status"]][0], 120),
        kpi("💰", "Estimated cost", inr(A["est_cost"]), f'materials ≈ {inr(A["cost_det"])}', TEAL, 180),
    ])
    P = build_payload(A)
    try:
        with st.spinner("📄 Building PDF…"):
            pdf_bytes = build_pdf(json.dumps(P, default=_json_default, sort_keys=True), ss.img_bytes or b"")
        csv_df = pd.concat(
            [pd.DataFrame({"Drawing file": [str(ss.img_name)] * len(A["comp_df"])}),
             A["comp_df"].rename(columns={"Drawing": "Drawing (detected)"}).reset_index(drop=True)], axis=1)
        csv_df["Overall deviation (%)"] = A["overall_dev"]
        csv_df["Tolerance (%)"] = A["tol"]
        csv_df["Status"] = A["status"]
        section("⬇️", "Download", "Both files are generated from the current live inputs")
        d1, d2 = st.columns(2)
        d1.download_button("📥 Download PDF report", data=pdf_bytes, file_name="validation_report.pdf",
                           mime="application/pdf", key="dl_pdf")
        d2.download_button("📊 Download summary CSV", data=csv_df.to_csv(index=False).encode("utf-8"),
                           file_name="validation_summary.csv", mime="text/csv", key="dl_csv")
    except Exception:
        callout("high", "The PDF could not be generated",
                "Please check that the `fpdf2` package is installed (pip install fpdf2). Details are below.")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())

    section("👀", "Report preview", "What the PDF contains")
    table(A["comp_df"])
    for sev, t, x in A["recs"]:
        callout(sev, t, x)
    if len(A["room_mat"]):
        plot(fig_room_materials(A["room_mat"]), "rep_roommat")
    footer()


# ---------------------------------------------------------
# PAGE: HISTORY
# ---------------------------------------------------------
def render_history():
    hero("Validation History", "Every saved validation of this session - search, filter, chart and download.", "History")
    if not ss.history:
        empty_state("📜", "No history yet", "Run a validation and press ‘Save to History’ on the Validate page.")
        st.write("")
        c1, c2, c3 = st.columns([1, 1.2, 1])
        c2.button("📐 Go to Validate Drawing", on_click=goto, args=(PAGE_VAL,), key="his_go")
        footer()
        return
    df = pd.DataFrame(ss.history)
    ok = (df["Status"] == "Valid").sum()
    kpi_grid([
        kpi("📜", "Saved", str(len(df)), "validations", VIOLET),
        kpi("✅", "Valid", str(int(ok)), f"{ok / len(df) * 100:.0f}% of saved", EMERALD, 60),
        kpi("📉", "Average deviation", f'{df["Deviation (%)"].mean():.1f}%', "across saved runs", ORANGE, 120),
        kpi("🏆", "Best deviation", f'{df["Deviation (%)"].min():.1f}%', "lowest saved", TEAL, 180),
    ])
    f1, f2 = st.columns([1.4, 1])
    f1.text_input("🔎 Search drawing / status", key="hist_q")
    f2.multiselect("Status filter", ["Valid", "Minor Difference", "Invalid"], key="hist_f")
    view = df[df["Status"].isin(ss.hist_f)]
    if ss.hist_q.strip():
        q = ss.hist_q.strip().lower()
        view = view[view.apply(lambda r: q in " ".join(map(str, r.values)).lower(), axis=1)]
    table(view)
    if len(view):
        c1, c2 = st.columns([1.6, 1], gap="large")
        with c1:
            plot(fig_history(view.reset_index(drop=True)), "his_trend")
        with c2:
            plot(fig_status_donut(view), "his_donut")
    b1, b2 = st.columns(2)
    b1.download_button("📥 Download history CSV", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="validation_history.csv", mime="text/csv", key="his_dl")
    if b2.button("🗑️ Clear history", key="his_clear"):
        ss.history = []
        ss.last_saved_sig = None
        st.rerun()
    footer()


# ---------------------------------------------------------
# PAGE: ABOUT
# ---------------------------------------------------------
def render_about():
    hero("About this system", "Validate construction estimates against 2D floor plans - for civil engineers, QS and architects.",
         "About")
    section("🎯", "Objective")
    st.write("The system reads a floor-plan image (OCR + OpenCV), derives plan size, rooms, walls and openings, "
             "compares them to your own estimate, classifies the result with rule-based and ML checks, and turns "
             "the drawing into a material take-off and a downloadable report.")
    section("🔁", "Workflow")
    st.code("Upload drawing → OCR (dimensions + room labels) → Calibrate units → Editable rooms table\n"
            "      → Live validation (rule engine + Random Forest) → Room-wise & bulk materials\n"
            "      → Recommendation → History / PDF / CSV", language="text")
    section("🧠", "How values are derived (assumptions)")
    st.markdown(f"""
- **Plan area** = overall width × depth read from the two large numbers near the outer walls (or manual entry).
- **Wall length** = (sum of room perimeters + outer perimeter) / 2 when room sizes are known, otherwise outer perimeter × 1.5.
- **Doors / windows / columns** are *rule-based expectations*, not symbol detection: doors = rooms + 1, windows = rooms, columns on a ≤ {COLUMN_SPACING:.0f} m grid.
- **Bricks** = net wall area × wall thickness ÷ {BRICK_VOL} m³ per brick (mortar included); openings deduct {DOOR_AREA} m² per door and {WINDOW_AREA} m² per window.
- **Concrete**: columns = nos × size² × height; slab = area × thickness. **Plaster** {PLASTER_T * 1000:.0f} mm thick, ≈ {BAGS_PER_M3} cement bags per m³ of mortar.
- **Tiles** 600×600 mm + 5 % wastage. **Paint** = wall area ÷ {PAINT_COVER:.0f} m² per litre per coat × {PAINT_COATS} coats (1 door + 1 window deducted per room).
- **Status**: Valid if the worst deviation ≤ tolerance, Minor Difference if ≤ 1.5 × tolerance, otherwise Invalid.
- **Rates** are editable placeholders - not market prices.
""")
    section("🛠️", "Technology")
    st.markdown("Streamlit · OpenCV · Tesseract OCR · scikit-learn (Random Forest) · Plotly · FPDF2 · Pillow")
    section("📸", "Drawing guidelines for OCR")
    st.markdown(GUIDE_MD)
    footer()


# ---------------------------------------------------------
# SIDEBAR  (rendered last so status is always fresh)
# ---------------------------------------------------------
def render_sidebar(A):
    with st.sidebar:
        html('<div class="logo"><div class="logo-ic">📐</div><div><div class="logo-t">EDV System</div>'
             '<div class="logo-s">Drawing Estimation Validation</div></div></div>')
        st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
        mdl = "✅ Ready" if MODELS["error"] is None else "⚠️ Missing"
        ocr = "✅ Ready" if tesseract_ok() else "❌ Not found"
        drw = "✅ Loaded" if ss.img_bytes else "— None"
        stat = A["status"] if A else "—"
        html(f'<div class="sb-card"><div class="sb-row"><span>ML models</span><b>{mdl}</b></div>'
             f'<div class="sb-row"><span>OCR engine</span><b>{ocr}</b></div>'
             f'<div class="sb-row"><span>Drawing</span><b>{drw}</b></div>'
             f'<div class="sb-row"><span>Live status</span><b>{stat}</b></div>'
             f'<div class="sb-row"><span>Saved</span><b>{len(ss.history)}</b></div></div>')
        if st.button("🗑️ Clear History", key="sb_clear"):
            ss.history = []
            ss.last_saved_sig = None
            st.toast("History cleared")
        html('<div class="logo-s" style="margin-top:.8rem;opacity:.75">BlueprintX UI · v3.1</div>')


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
inject_css()
_page = ss.page
if _page == PAGE_VAL:
    _A = render_validate()
else:
    _A = compute_analysis()
    if _page == PAGE_DASH:
        render_dashboard(_A)
    elif _page == PAGE_MAT:
        render_materials(_A)
    elif _page == PAGE_REP:
        render_report(_A)
    elif _page == PAGE_HIS:
        render_history()
    else:
        render_about()
render_sidebar(_A)
