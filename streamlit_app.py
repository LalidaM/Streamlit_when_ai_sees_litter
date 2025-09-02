import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from ultralytics import YOLO

# ======================= Streamlit Config =======================
st.set_page_config(page_title="When AI Sees Litter", page_icon="♻️", layout="wide")

# ======================= THEME (Light agriculture vibe) =======================
def apply_agri_theme():
    st.markdown("""
    <style>
      :root{
        --agri-primary:#79C16D;
        --agri-primary-dark:#4FA25A;
        --agri-accent:#CFEAC0;
        --agri-bg:#FAFEF6;
        --agri-card:#FFFFFF;
        --agri-text:#0F2A1C;
        --agri-muted:#6F8B7A;
        --agri-pill:#EEF7E9;
        --agri-border:#E5EFE3;
      }
      html, body, [data-testid="stAppViewContainer"]{
        background: var(--agri-bg);
        color: var(--agri-text);
      }
      .main .block-container{ padding-top: 1rem !important; }
      .hero{ background:
          radial-gradient(700px 280px at 10% -20%, rgba(121,193,109,.18), transparent),
          linear-gradient(135deg, #FFFFFF 0%, #F7FBF2 100%);
        border:1px solid var(--agri-border);
        border-radius:24px; padding:22px 20px; margin: 6px 0 12px 0;
      }
      .hero h1{ margin:0 0 6px 0; font-weight:900; font-size:1.8rem; }
      .hero p{ margin:0 0 10px 0; color: var(--agri-muted); }
      .pill{ display:inline-block; background: var(--agri-pill); padding: 2px 10px;
             border-radius: 999px; color: var(--agri-primary-dark); border:1px solid var(--agri-border); }
      .section{ margin: 10px 0 18px 0; padding:16px; background: var(--agri-card);
                border:1px solid var(--agri-border); border-radius: 18px; }
      .eco-card{ background:#FFFFFF; border:1px solid var(--agri-border); border-radius:22px;
                 padding:18px 16px; margin:10px 0 18px 0; box-shadow: 0 3px 16px rgba(0,0,0,.04); }
      .eco-head{ display:flex; align-items:center; gap:10px; margin-bottom:6px; }
      .eco-emoji{ font-size:1.5rem; }
      .eco-title{ font-weight:900; font-size:1.28rem; }
      .eco-badge{ margin-left:auto; background: var(--agri-pill); color: var(--agri-primary-dark);
                  border:1px solid var(--agri-border); border-radius:999px; padding:4px 10px; font-size:.85rem; }
      .eco-meta{ margin: 6px 0 8px 0; color: var(--agri-muted); font-size:.95rem; }
      .eco-section-title{ font-weight:800; margin-top:8px; margin-bottom:4px; }
      .eco-list{ margin:0 0 4px 0; padding-left:18px;}
      .eco-list li{ margin: 2px 0; }
      .chip-row{ display:flex; flex-wrap:wrap; gap:8px; margin: 6px 0 2px 0; }
      .chip{ background: var(--agri-pill); color: var(--agri-primary-dark); border:1px solid var(--agri-border);
             border-radius:999px; padding:4px 10px; font-size:.88rem; }
      .eco-links{ display:flex; gap:10px; margin-top:10px; flex-wrap:wrap; }
      .eco-link{ border-radius:999px; padding:8px 12px; border:1px solid var(--agri-border);
                 background:#fff; text-decoration:none !important; color: var(--agri-primary-dark) !important; font-weight:700; }
      .eco-link:hover{ background: var(--agri-pill); }
      .citybadge{ display:inline-block; background: var(--agri-pill); padding:4px 10px;
                  border-radius:999px; border:1px solid var(--agri-border); color: var(--agri-primary-dark); }
      .sdg-row{ display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
      .sdg-card{ display:flex; gap:10px; align-items:center; border:1px solid var(--agri-border);
                 background:#fff; padding:10px 12px; border-radius:14px; }
      .sdg-card img{ width:54px; height:auto; }
      .sdg-card .txt{ font-weight:700; }
      .link-chips{ display:flex; flex-wrap:wrap; gap:10px; margin-top:8px; }
      .link-chip{ border:1px solid var(--agri-border); padding:8px 12px; border-radius:999px;
                  text-decoration:none; color: var(--agri-primary-dark); background:#fff; font-weight:700; }
      .link-chip:hover{ background: var(--agri-pill); }
    </style>
    """, unsafe_allow_html=True)

apply_agri_theme()

# ======================= Config & Model =======================
MODEL_URL     = os.getenv("MODEL_URL", "https://raw.githubusercontent.com/Bellzum/streamlit-main/main/yolo_litterv1.pt")
LOCAL_MODEL   = os.getenv("LOCAL_MODEL", "best.pt")
CACHED_PATH   = "/tmp/models/best.pt"
DEFAULT_IMGSZ = int(os.getenv("IMGSZ", "640"))

CLASS_NAMES   = ["Clear plastic bottle", "Drink can", "Plastic bottle cap"]
IMGSZ_OPTIONS = [320, 416, 512, 640, 800, 960, 1280]

# SDG icons
SDG_11 = "https://sdgs.un.org/sites/default/files/2020-09/SDG-11.png"
SDG_12 = "https://sdgs.un.org/sites/default/files/2020-09/SDG-12.png"
SDG_13 = "https://sdgs.un.org/sites/default/files/2020-09/SDG-13.png"
SDG_14 = "https://sdgs.un.org/sites/default/files/2020-09/SDG-14.png"

# ======================= Helpers =======================
def _download_file(url: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        import requests
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk: f.write(chunk)
    except Exception:
        import urllib.request
        with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)

def _ensure_model_path() -> str:
    if MODEL_URL.strip().startswith("http"):
        if not os.path.exists(CACHED_PATH):
            _download_file(MODEL_URL.strip(), CACHED_PATH)
        return CACHED_PATH
    if not os.path.exists(LOCAL_MODEL):
        st.error("Model file not found.")
        st.stop()
    return LOCAL_MODEL

def _cache_key_for(path: str) -> str:
    try: return f"{path}:{os.path.getmtime(path)}:{os.path.getsize(path)}"
    except Exception: return path

@st.cache_resource(show_spinner=True)
def _load_model_cached(path: str, key: str):
    return YOLO(path)

def load_model():
    path = _ensure_model_path()
    return _load_model_cached(path, _cache_key_for(path))

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"))[:, :, ::-1]

# --- NEW: draw_boxes using PIL instead of cv2
def draw_boxes(bgr, dets):
    img = Image.fromarray(bgr[:, :, ::-1])
    draw = ImageDraw.Draw(img)
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        label = f'{d["class_name"]} {d["score"]:.2f}'
        draw.rectangle([x1, y1, x2, y2], outline=(28,160,78), width=2)
        draw.text((x1, max(0, y1-12)), label, fill=(28,160,78))
    return img

def _closest_size(target: int, options: list[int]) -> int:
    return min(options, key=lambda x: abs(x - target))

# ======================= UI Header =======================
logo_col, title_col = st.columns([3, 5])
with logo_col:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
with title_col:
    st.markdown("<div style='font-weight:800; font-size:1.6rem;'>When AI Sees Litter </div>", unsafe_allow_html=True)

# ======================= QUICK DETECT =======================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("#### How to use")
st.markdown("""
1. Upload or capture an image.
2. Run detection.
3. Follow disposal steps for your city.
""")

# Default thresholds
conf = 0.05; iou = 0.10; imgsz = _closest_size(DEFAULT_IMGSZ, IMGSZ_OPTIONS)
bottle_min = 0.00; can_min = 0.00; cap_min = 0.00; min_area_pct = 0.0; tta = False

with st.expander("Advanced settings (optional)"):
    preset = st.radio("Preset", ["Minimum filters", "Recommended", "Strict"], index=0, horizontal=True)
    if preset == "Recommended":
        conf = 0.25; iou = 0.45; bottle_min = 0.60; can_min = 0.55; cap_min = 0.65; min_area_pct = 0.3
    elif preset == "Strict":
        conf = 0.35; iou = 0.50; bottle_min = 0.70; can_min = 0.70; cap_min = 0.75; min_area_pct = 0.5
    conf = st.slider("Base confidence", 0.05, 0.95, conf, 0.01)
    iou  = st.slider("IoU", 0.10, 0.90, iou, 0.01)
    imgsz = st.select_slider("Inference size", options=IMGSZ_OPTIONS, value=imgsz)

# Input controls
src = st.radio("Input source", ["Upload image", "Camera"], index=0, horizontal=True)
image = None
if src == "Upload image":
    up = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if up: image = Image.open(up).convert("RGB")
else:
    shot = st.camera_input("Open your camera", key="cam1")
    if shot: image = Image.open(shot).convert("RGB")

# Model loader
if st.button("Load model"):
    m = load_model()
    st.success("Model ready.")

# Run detection
if image is not None:
    st.image(image, caption="Input", use_container_width=True)
    if st.button("Run detection"):
        model = load_model()
        bgr = pil_to_bgr(image)
        results = model.predict(bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False, augment=tta)
        pred = results[0]

        if pred.boxes is None or len(pred.boxes) == 0:
            st.info("No detections")
        else:
            boxes  = pred.boxes.xyxy.cpu().numpy()
            scores = pred.boxes.conf.cpu().numpy()
            clsi   = pred.boxes.cls.cpu().numpy().astype(int)
            names_map = pred.names if hasattr(pred, "names") else model.names

            dets = []
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].tolist()
                c = int(clsi[i])
                name = names_map.get(c, str(c)) if isinstance(names_map, dict) else CLASS_NAMES[c]
                s = float(scores[i])
                dets.append({"xyxy": [x1, y1, x2, y2], "class_id": c, "class_name": name, "score": s})

            if not dets:
                st.info("Filtered out all detections.")
            else:
                vis_img = draw_boxes(bgr, dets)
                st.subheader("Detections")
                st.image(vis_img, use_container_width=True)
                with st.expander("Raw detections", expanded=False):
                    st.dataframe(pd.DataFrame(dets))

st.markdown('</div>', unsafe_allow_html=True)
