import io
import os
import json
import base64
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import streamlit as st

import sqlite3
from datetime import datetime

# ---------------------------
# App-wide constants & styles
# ---------------------------
APP_NAME = "Breast Imaging AI Workbench"
PRIMARY = "#e91e63"  # rose/pink — breast cancer awareness
ACCENT = "#9c27b0"   # deep purple as secondary accent

CUSTOM_CSS = f"""
<style>
:root {{
  --primary: {PRIMARY};
  --accent: {ACCENT};
}}
.block-container {{ padding-top: 1.1rem; padding-bottom: 1.1rem; }}
.tile {{
  border: 1px solid #eee; border-radius: 16px; padding: 18px; text-align:center;
  box-shadow: 0 4px 14px rgba(0,0,0,0.06);
  transition: transform .08s ease-in-out, box-shadow .08s ease-in-out;
}}
.tile:hover {{ transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,0,0,0.08); }}
.tile h3 {{ margin: 0.2rem 0 0.2rem 0; }}
</style>
"""

# ---------------------------
# Data structures
# ---------------------------
@dataclass
class ModelEntry:
    id: Optional[int]
    name: str
    modality: str  # Mammography, Ultrasound, MRI, PET, Pathology, Tabular
    task: str      # e.g., BIRADS, Density, Mass/Calc Seg, Benign/Malignant, etc.
    framework: str # keras|pytorch|onnx|yolo|other
    file_name: Optional[str] = None
    disk_path: Optional[str] = None
    # Optional metadata
    yolo_task: Optional[str] = None
    class_names: Optional[List[str]] = None
    input_size: Optional[Tuple[int, int]] = None
    label_map: Optional[List[str]] = None

# ---------------------------
# Session State Helpers
# ---------------------------
BASE_DIR = "breast_ai_workbench"
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_DIR = os.path.join(BASE_DIR, "db")
IMG_DIR = os.path.join(BASE_DIR, "img") ### EKLENDİ ###

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True) ### EKLENDİ ###

def init_state():
    if "results" not in st.session_state:
        st.session_state.results = []
    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = ""
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    # runtime
    if "device" not in st.session_state:
        st.session_state.device = "auto"
    if "yolo_conf" not in st.session_state:
        st.session_state.yolo_conf = 0.25
    if "yolo_iou" not in st.session_state:
        st.session_state.yolo_iou = 0.45

    # theme
    if "theme_primary" not in st.session_state:
        st.session_state.theme_primary = PRIMARY
    if "theme_accent" not in st.session_state:
        st.session_state.theme_accent = ACCENT
    if "theme_palette" not in st.session_state:
        st.session_state.theme_palette = "Pink/Purple"
    if "compact" not in st.session_state:
        st.session_state.compact = False

    # patient/case
    if "current_case" not in st.session_state:
        st.session_state.current_case = {}
    if "current_case_id" not in st.session_state:
        st.session_state.current_case_id = None

    # database
    if "db_path" not in st.session_state:
        st.session_state.db_path = os.path.join(DB_DIR, "breast_ai_workbench.db")

init_state()

st.set_page_config(page_title=APP_NAME, page_icon="🎗️", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def apply_runtime_theme():
    prim = st.session_state.get("theme_primary", PRIMARY)
    acc = st.session_state.get("theme_accent", ACCENT)
    compact_css = (
        ".block-container{padding-top:.6rem;padding-bottom:1rem;}"
        ".tile{padding:14px;}"
        ".stDataFrame, .stDataEditor{font-size:13px;}"
    ) if st.session_state.get("compact") else ""
    st.markdown(f"<style>:root{{--primary:{prim};--accent:{acc};}} {compact_css}</style>", unsafe_allow_html=True)

apply_runtime_theme()

# ---------------------------
# Utilities
# ---------------------------
MODALITIES = [
    ("Mammography", "BIRADS, Density, Mass & Calcification"),
    ("Ultrasound", "BIRADS, Malignant/Benign/Normal, Segmentation"),
    ("MRI", "(bring your own models)"),
    ("PET", "(bring your own models)"),
    ("Tabular", "Wisconsin Diagnostic dataset"),
    ("Pathology", "WSI/Patches, basic hooks"),
]

TASKS_BY_MODALITY = {
    "Mammography": ["BIRADS", "Density", "Mass&Calc Seg"],
    "Ultrasound": ["BIRADS", "Malignant/Benign/Normal", "Segmentation"],
    "MRI": ["Classification", "Segmentation"],
    "PET": ["Classification", "Segmentation"],
    "Pathology": ["Classification", "Segmentation"],
    "Tabular": ["Wisconsin Predict"],
}

FRAMEWORKS = ["keras", "pytorch", "onnx", "yolo", "other"]

def save_model_file(uploaded_file) -> str:
    """Save uploaded model file into persistent MODELS_DIR and return absolute path."""
    safe = "".join(c if c.isalnum() or c in ('.', '_') else "_" for c in uploaded_file.name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{ts}_{safe}"
    path = os.path.join(MODELS_DIR, fname)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

### EKLENDİ ###
def save_image_file(uploaded_file) -> str:
    """Save uploaded image file into persistent IMG_DIR and return absolute path."""
    safe = "".join(c if c.isalnum() or c in ('.', '_') else "_" for c in uploaded_file.name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{ts}_{safe}"
    path = os.path.join(IMG_DIR, fname)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

# ---------------------------
# SQLite persistence
# ---------------------------
def db_connect():
    return sqlite3.connect(st.session_state.db_path, check_same_thread=False)

def db_init():
    con = db_connect(); cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            modality TEXT NOT NULL,
            task TEXT NOT NULL,
            framework TEXT NOT NULL,
            file_name TEXT,
            file_path TEXT,
            class_names TEXT,  -- JSON
            input_size TEXT,   -- JSON [w,h]
            label_map TEXT,    -- JSON
            created_at TEXT,
            active INTEGER DEFAULT 1
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            patient_name TEXT,
            age INTEGER,
            sex TEXT,
            study_date TEXT,
            modality TEXT,
            laterality TEXT,
            view TEXT,
            notes TEXT,
            created_at TEXT
        )
    """)
    ### DEĞİŞİKLİK: image_path sütunu eklendi ###
    cur.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id INTEGER,
            model_name TEXT,
            file_name TEXT,
            image_path TEXT,
            task TEXT,
            ok INTEGER,
            label TEXT,
            score REAL,
            predictions_json TEXT,
            overlay_b64 TEXT,
            created_at TEXT,
            FOREIGN KEY(case_id) REFERENCES cases(id)
        )
    """)
    con.commit(); con.close()

db_init()

# Model CRUD helpers
def add_model_to_db(entry: ModelEntry) -> int:
    con = db_connect(); cur = con.cursor()
    cur.execute("""
        INSERT INTO models (name, modality, task, framework, file_name, file_path, class_names, input_size, label_map, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        entry.name, entry.modality, entry.task, entry.framework,
        entry.file_name, entry.disk_path,
        json.dumps(entry.class_names) if entry.class_names else None,
        json.dumps(list(entry.input_size)) if entry.input_size else None,
        json.dumps(entry.label_map) if entry.label_map else None,
        datetime.utcnow().isoformat()
    ))
    mid = cur.lastrowid; con.commit(); con.close()
    return mid

def get_models_from_db(modality: Optional[str] = None, task: Optional[str] = None) -> List[ModelEntry]:
    con = db_connect()
    q = "SELECT * FROM models WHERE active=1"
    params = []
    if modality:
        q += " AND modality=?"; params.append(modality)
    if task:
        q += " AND task=?"; params.append(task)
    q += " ORDER BY created_at DESC"
    df = pd.read_sql_query(q, con, params=params); con.close()
    out: List[ModelEntry] = []
    for _, r in df.iterrows():
        out.append(ModelEntry(
            id=int(r["id"]),
            name=str(r["name"]), modality=str(r["modality"]), task=str(r["task"]), framework=str(r["framework"]),
            file_name=r["file_name"], disk_path=r["file_path"],
            class_names=json.loads(r["class_names"]) if r["class_names"] else None,
            input_size=tuple(json.loads(r["input_size"])) if r["input_size"] else None,
            label_map=json.loads(r["label_map"]) if r["label_map"] else None
        ))
    return out

def db_insert_case(meta: Dict[str, Any]) -> int:
    con = db_connect(); cur = con.cursor()
    cur.execute("""INSERT INTO cases
        (patient_id, patient_name, age, sex, study_date, modality, laterality, view, notes, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (meta.get("patient_id"), meta.get("patient_name"), meta.get("age"), meta.get("sex"),
         meta.get("study_date"), meta.get("modality"), meta.get("laterality"), meta.get("view"),
         meta.get("notes"), datetime.utcnow().isoformat()))
    cid = cur.lastrowid; con.commit(); con.close()
    return cid

### DEĞİŞİKLİK: Fonksiyon imzası ve INSERT sorgusu hatayı gidermek ve yeni alanları eklemek için güncellendi ###
def db_insert_result(case_id: int, r: Dict[str, Any]):
    label, score = None, None
    preds = r.get("predictions") or []
    if preds and isinstance(preds, list) and isinstance(preds[0], dict):
        p0 = preds[0]
        label = p0.get("label")
        try:
            score = float(p0.get("score")) if p0.get("score") is not None else None
        except Exception:
            score = None
    con = db_connect(); cur = con.cursor()
    cur.execute("""INSERT INTO results
        (case_id, model_name, file_name, image_path, task, ok, label, score, predictions_json, overlay_b64, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (case_id, r.get("model"), r.get("file"), r.get("image_path"), r.get("task"),
         int(bool(r.get("ok"))), label, score,
         json.dumps(r.get("predictions", [])), r.get("overlay_b64") or r.get("boxed_b64"),
         datetime.utcnow().isoformat()))
    con.commit(); con.close()

def db_fetch_results(filters: Dict[str, Any] = None) -> pd.DataFrame:
    con = db_connect()
    ### DEĞİŞİKLİK: image_path sütunu sorguya eklendi ###
    q = (
        "SELECT results.id, results.created_at, results.file_name, results.image_path, results.task, results.ok, "
        "results.label, results.score, results.model_name, "
        "cases.id AS case_id, cases.patient_id, cases.patient_name, cases.age, cases.sex, cases.study_date, "
        "cases.modality, cases.laterality, cases.view, cases.notes "
        "FROM results "
        "JOIN cases ON results.case_id = cases.id "
    )
    params = []
    conds = []
    if filters:
        if filters.get("patient_query"):
            conds.append("(cases.patient_id LIKE ? OR cases.patient_name LIKE ?)")
            pq = f"%{filters['patient_query']}%"; params += [pq, pq]
        if filters.get("modality") and filters["modality"] != "All":
            conds.append("cases.modality = ?"); params.append(filters["modality"])
    if conds:
        q += " WHERE " + " AND ".join(conds)
    q += " ORDER BY datetime(results.created_at) DESC"
    df = pd.read_sql_query(q, con, params=params)
    con.close()
    return df

def db_update_case_notes(patient_id: str, notes: str):
    con = db_connect(); cur = con.cursor()
    cur.execute("UPDATE cases SET notes=? WHERE patient_id=?", (notes, patient_id))
    con.commit(); con.close()

# ---------------------------
# Inference helpers
# ---------------------------
def ensure_image(img_or_bytes) -> Image.Image:
    if isinstance(img_or_bytes, Image.Image):
        return img_or_bytes
    return Image.open(io.BytesIO(img_or_bytes)).convert("RGB")

def draw_boxes_overlay(base_img: Image.Image, boxes: List[Dict[str, Any]]) -> str:
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    for b in boxes:
        try:
            x1,y1,x2,y2 = [int(v) for v in b.get("bbox", [])]
            draw.rectangle([x1,y1,x2,y2], outline=(233,30,99), width=3)
            lbl = f"{b.get('label','')} {b.get('score',0):.2f}" if b.get("score") is not None else b.get("label","")
            if lbl:
                draw.text((x1+4, max(0,y1-14)), lbl, fill=(233,30,99))
        except Exception:
            continue
    bio = io.BytesIO(); img.save(bio, format="PNG")
    return base64.b64encode(bio.getvalue()).decode("utf-8")

# ---------------------------
# Inference dispatch (skeleton)
# ---------------------------
def run_inference(model: ModelEntry, image_bytes: bytes) -> Dict[str, Any]:
    """Route to the correct backend. Returns a dict with standardized fields."""
    img = ensure_image(image_bytes)

    try:
        if model.framework == "yolo":
            return infer_yolo(model, img)
        elif model.framework == "keras":
            return infer_keras(model, img)
        elif model.framework == "pytorch":
            return infer_torch(model, img)
        elif model.framework == "onnx":
            return infer_onnx(model, img)
        else:
            raise RuntimeError("Unsupported framework; returning stub result.")
    except Exception as e:
        return {"model": model.name, "modality": model.modality, "task": model.task, "ok": False, "error": str(e), "predictions": []}

# ---- YOLO / Keras / Torch / ONNX backends (with graceful fallbacks)
def infer_yolo(model: ModelEntry, img: Image.Image) -> Dict[str, Any]:
    try:
        from ultralytics import YOLO  # type: ignore
        import torch  # noqa: F401
    except Exception:
        boxes = [{"label": "lesion", "bbox": [50,50,150,150], "score": 0.81}]
        return {"model": model.name, "modality": model.modality, "task": model.task, "ok": True,
                "note": "ultralytics/torch not installed; mock result",
                "predictions": boxes, "boxed_b64": draw_boxes_overlay(img, boxes)}

    if not model.disk_path:
        raise RuntimeError("YOLO model missing disk_path")
    ymodel = YOLO(model.disk_path)
    device = st.session_state.get("device", "auto")
    device_arg = None if device == "auto" else device
    conf = float(st.session_state.get("yolo_conf", 0.25))
    iou = float(st.session_state.get("yolo_iou", 0.45))

    res = ymodel.predict(img, device=device_arg, conf=conf, iou=iou, verbose=False)

    preds = []
    class_names = model.class_names
    overlay_b64 = None
    boxed_b64 = None

    try:
        r0 = res[0]
        boxes = []
        if hasattr(r0, 'boxes') and r0.boxes is not None:
            for b in r0.boxes:
                cls_id = int(b.cls[0].item()) if hasattr(b.cls[0], 'item') else int(b.cls[0])
                confv = float(b.conf[0]) if hasattr(b.conf[0], '__float__') else float(b.conf[0])
                xyxy = b.xyxy[0].tolist()
                label = class_names[cls_id] if class_names and 0 <= cls_id < len(class_names) else str(cls_id)
                item = {"label": label, "bbox": xyxy, "score": confv}
                preds.append(item); boxes.append(item)
        if boxes:
            boxed_b64 = draw_boxes_overlay(img, boxes)
        if hasattr(r0, 'masks') and r0.masks is not None and r0.masks.data is not None:
            try:
                mdata = r0.masks.data
                arr = np.array(img).copy()
                for mi in range(mdata.shape[0]):
                    mask = mdata[mi].cpu().numpy() > 0.5
                    arr[mask] = (0.6 * np.array([233, 30, 99]) + 0.4 * arr[mask]).astype(np.uint8)
                ov = Image.fromarray(arr); bio = io.BytesIO(); ov.save(bio, format='PNG')
                overlay_b64 = base64.b64encode(bio.getvalue()).decode('utf-8')
            except Exception:
                pass
    except Exception:
        pass

    out = {"model": model.name, "modality": model.modality, "task": model.task, "ok": True, "predictions": preds}
    if overlay_b64: out["overlay_b64"] = overlay_b64
    if boxed_b64: out["boxed_b64"] = boxed_b64
    return out

def infer_keras(model: ModelEntry, img: Image.Image) -> Dict[str, Any]:
    try:
        import tensorflow as tf  # type: ignore
        import numpy as _np
    except Exception:
        return {"model": model.name, "modality": model.modality, "task": model.task, "ok": True,
                "note": "tensorflow not installed; mock softmax",
                "predictions": [{"label": "BIRADS-3", "score": 0.62}]}

    if not model.disk_path:
        raise RuntimeError("Keras model missing disk_path")

    kmodel = tf.keras.models.load_model(model.disk_path)
    size = model.input_size or (224, 224)
    arr = _np.array(img.resize(size)) / 255.0
    arr = _np.expand_dims(arr, axis=0)
    y = kmodel.predict(arr, verbose=0)

    if y.ndim == 2 and y.shape[1] <= 64:
        probs = y[0].tolist()
        labels = model.label_map or [f"class_{i}" for i in range(len(probs))]
        idx = int(_np.argmax(probs))
        return {"model": model.name, "modality": model.modality, "task": model.task, "ok": True,
                "predictions": [{"label": labels[idx], "score": float(probs[idx])}]}

    return {"model": model.name, "modality": model.modality, "task": model.task, "ok": True,
            "predictions": [{"vector": y[0].tolist()}]}

def infer_torch(model: ModelEntry, img: Image.Image) -> Dict[str, Any]:
    try:
        import torch  # type: ignore
        import numpy as _np
    except Exception:
        return {"model": model.name, "modality": model.modality, "task": model.task, "ok": True,
                "note": "torch not installed; mock result",
                "predictions": [{"label": "benign", "score": 0.71}]}

    if not model.disk_path:
        raise RuntimeError("Torch model missing disk_path")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = torch.jit.load(model.disk_path, map_location=device) if model.disk_path.endswith(".pt") else torch.load(model.disk_path, map_location=device)
    m.eval()
    with torch.no_grad():
        tens = torch.from_numpy(_np.array(img.resize((224,224))) / 255.0).float().permute(2,0,1).unsqueeze(0).to(device)
        y = m(tens)
    if hasattr(y, "softmax"):
        y = y.softmax(dim=1)
    if isinstance(y, (list, tuple)):
        y = y[0]
    if hasattr(y, "cpu"):
        y = y.cpu().numpy()
    probs = y[0] if y.ndim == 2 else y
    idx = int(np.argmax(probs))
    return {"model": model.name, "modality": model.modality, "task": model.task, "ok": True,
            "predictions": [{"label": f"class_{idx}", "score": float(probs[idx])}]}

def infer_onnx(model: ModelEntry, img: Image.Image) -> Dict[str, Any]:
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        return {"model": model.name, "modality": model.modality, "task": model.task, "ok": True,
                "note": "onnxruntime not installed; mock result",
                "predictions": [{"label": "density_C", "score": 0.55}]}

    if not model.disk_path:
        raise RuntimeError("ONNX model missing disk_path")

    sess = ort.InferenceSession(model.disk_path, providers=["CPUExecutionProvider"])  # add CUDA if available
    input_name = sess.get_inputs()[0].name
    arr = np.array(img.resize((224,224)), dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2,0,1))[None, ...]
    y = sess.run(None, {input_name: arr})[0]
    probs = y[0] if y.ndim == 2 else y
    idx = int(np.argmax(probs))
    return {"model": model.name, "modality": model.modality, "task": model.task, "ok": True,
            "predictions": [{"label": f"class_{idx}", "score": float(probs[idx])}]}

# ---------------------------
# Reporting & Gemini polish
# ---------------------------
def summarize_results(results: List[Dict[str, Any]]) -> str:
    if not results: return "No results yet."
    lines = [f"# {APP_NAME} — Report\n"]
    for r in results:
        lines.append(f"**Model**: {r.get('model')}  ")
        lines.append(f"**Modality**: {r.get('modality')}  ")
        lines.append(f"**Task**: {r.get('task')}  ")
        if r.get("ok"):
            preds = r.get("predictions", [])
            if preds:
                lines.append("**Predictions:**")
                for p in preds[:10]:
                    if isinstance(p, dict) and "label" in p and "score" in p:
                        try: lines.append(f"- {p['label']} (score={float(p['score']):.2f})")
                        except Exception: lines.append(f"- {p['label']}")
                    else:
                        lines.append(f"- {json.dumps(p)[:200]}")
            else:
                lines.append("(No predictions)")
        else:
            lines.append(f"❗ Error: {r.get('error')}")
        lines.append("")
    return "\n".join(lines)

def polish_with_gemini(text: str, api_key: str) -> str:
    if not api_key: return text
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(
            "You are an expert medical report editor. Rewrite the following draft into a clear, professional, and concise clinician-friendly report. "
            "Keep all factual content, add brief plain-language explanations where helpful, and avoid overclaiming.\n\n" + text
        )
        return resp.text or text
    except Exception as e:
        return text + f"\n\n(Note: Gemini polish failed: {e})"

# ---------------------------
# UI building blocks
# ---------------------------
def header(title: str, subtitle: Optional[str] = None):
    st.markdown(f"<h1 style='margin-bottom:0.2rem'>{title}</h1>", unsafe_allow_html=True)
    if subtitle: st.caption(subtitle)

def tile(label: str, desc: str, key: str):
    with st.container():
        st.markdown(f"<div class='tile'><h3>{label}</h3><p style='min-height:36px'>{desc}</p>", unsafe_allow_html=True)
        go = st.button("Open", key=key)
        st.markdown("</div>", unsafe_allow_html=True)
        return go

def patient_meta_form(modality: str) -> Dict[str, Any]:
    with st.expander("Patient / Study metadata", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            pid = st.text_input("Patient ID", value=st.session_state.current_case.get("patient_id", ""))
            name = st.text_input("Name", value=st.session_state.current_case.get("patient_name", ""))
            age_val = st.session_state.current_case.get("age", 0) or 0
            age = st.number_input("Age", min_value=0, max_value=120, value=int(age_val))
        with c2:
            sex = st.selectbox("Sex", ["","F","M"], index=["","F","M"].index(st.session_state.current_case.get("sex","")) if st.session_state.current_case.get("sex","") in ["","F","M"] else 0)
            default_date = pd.to_datetime(st.session_state.current_case.get("study_date", pd.Timestamp.today())).date()
            study_date = st.date_input("Study date", value=default_date)
            laterality = st.selectbox("Laterality", ["","Left","Right","Bilateral"],
                                      index=["","Left","Right","Bilateral"].index(st.session_state.current_case.get("laterality","")) if st.session_state.current_case.get("laterality","") in ["","Left","Right","Bilateral"] else 0)
        with c3:
            view = st.selectbox("View", ["","CC","MLO","Other"],
                                index=["","CC","MLO","Other"].index(st.session_state.current_case.get("view","")) if st.session_state.current_case.get("view","") in ["","CC","MLO","Other"] else 0)
            notes = st.text_input("Notes", value=st.session_state.current_case.get("notes",""))
        if st.button("Set as current case", type="primary"):
            meta = {
                "patient_id": pid, "patient_name": name, "age": age, "sex": sex,
                "study_date": str(study_date), "modality": modality,
                "laterality": laterality, "view": view, "notes": notes
            }
            st.session_state.current_case = meta
            st.session_state.current_case_id = db_insert_case(meta)
            st.success(f"Case created for {pid or name} (ID: {st.session_state.current_case_id})")
    return st.session_state.current_case

# ---------------------------
# Pages
# ---------------------------
def page_home():
    header("🎗️ Breast Imaging AI Workbench", "Select a module to begin.")
    cols = st.columns(3)
    tiles = [
        ("Mammography", "BIRADS, Density, Mass & Calcification"),
        ("Ultrasound", "BIRADS, Malignant/Benign/Normal, Segmentation"),
        ("MRI", "Bring your own models"),
        ("PET", "Bring your own models"),
        ("Tabular", "Wisconsin Diagnostic dataset"),
        ("Pathology", "WSI/Patches"),
        ("Models", "Register and manage your models"),
        ("Reports", "Excel-like grid, filters, exports"),
        ("Settings", "Theme, Storage & Gemini API"),
    ]
    for i, (label, desc) in enumerate(tiles):
        with cols[i % 3]:
            if tile(label, desc, key=f"tile_{label}"):
                st.session_state.page = label

def page_settings():
    header("Settings", "Tokens, Theme & Storage")
    st.subheader("Gemini API Key")
    st.session_state.gemini_api_key = st.text_input("google-generativeai API key", value=st.session_state.gemini_api_key, type="password")
    st.info("Used only for on-demand report polishing. Stored in session memory, not on disk.")

    st.subheader("Theme")
    palette = st.selectbox("Palette preset", ["Pink/Purple","Teal/Indigo","Slate"],
                           index=["Pink/Purple","Teal/Indigo","Slate"].index(st.session_state.theme_palette))
    if palette != st.session_state.theme_palette:
        st.session_state.theme_palette = palette
        if palette == "Pink/Purple":
            st.session_state.theme_primary, st.session_state.theme_accent = "#e91e63", "#9c27b0"
        elif palette == "Teal/Indigo":
            st.session_state.theme_primary, st.session_state.theme_accent = "#009688", "#3f51b5"
        else:
            st.session_state.theme_primary, st.session_state.theme_accent = "#374151", "#6b7280"
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.theme_primary = st.color_picker("Primary color", value=st.session_state.theme_primary)
    with c2:
        st.session_state.theme_accent = st.color_picker("Accent color", value=st.session_state.theme_accent)
    st.session_state.compact = st.checkbox("Compact density", value=st.session_state.compact)
    apply_runtime_theme()

    st.subheader("Storage")
    st.session_state.db_path = st.text_input("SQLite DB path", value=st.session_state.db_path)
    if st.button("Reinitialize DB (keeps existing tables)"):
        try: db_init(); st.success("DB ensured.")
        except Exception as e: st.error(f"DB init failed: {e}")

def page_models():
    header("Model Registry", "Add/inspect models per modality & task")
    with st.expander("➕ Register a new model", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Model name", value="")
            modality = st.selectbox("Modality", [m for m,_ in MODALITIES])
            task = st.selectbox("Task", TASKS_BY_MODALITY.get(modality, ["Other"]))
            framework = st.selectbox("Framework", FRAMEWORKS)
        with c2:
            uploaded = st.file_uploader("Model file (.h5/.keras/.pt/.pth/.onnx)")
        class_names = None; input_size = None; label_map = None; yolo_task = None
        if framework == "yolo":
            yolo_task = st.selectbox("YOLO task", ["detect","segment"], index=0)
            names_text = st.text_input("Class names (comma-separated, optional)", value="")
            class_names = [s.strip() for s in names_text.split(",") if s.strip()] or None
        elif framework == "keras":
            c3, c4 = st.columns(2)
            with c3: iw = st.number_input("Input width", 32, 4096, 224)
            with c4: ih = st.number_input("Input height", 32, 4096, 224)
            input_size = (int(iw), int(ih))
            labels_text = st.text_input("Label map (comma-separated, optional)", value="")
            label_map = [s.strip() for s in labels_text.split(",") if s.strip()] or None

        if st.button("Add to registry", type="primary"):
            if not name or not uploaded:
                st.error("Please provide a model name and upload a file.")
            else:
                file_path = save_model_file(uploaded)
                entry = ModelEntry(
                    id=None, name=name, modality=modality, task=task, framework=framework,
                    file_name=uploaded.name, disk_path=file_path,
                    yolo_task=yolo_task, class_names=class_names, input_size=input_size, label_map=label_map
                )
                mid = add_model_to_db(entry)
                st.success(f"Added: {name} — saved at {file_path} (ID: {mid})")

    st.markdown("---")
    st.subheader("Registered Models")
    df = pd.DataFrame([asdict(m) for m in get_models_from_db()])
    if df.empty:
        st.info("No models saved yet.")
    else:
        st.dataframe(df[["id","name","modality","task","framework","file_name","disk_path"]], use_container_width=True)


def _modality_page(modality: str):
    header(modality)
    meta = patient_meta_form(modality)

    models = get_models_from_db(modality=modality)
    if not models:
        st.warning("No models registered yet for this modality. Add some in the Models page.")
    name_to_model = {m.name: m for m in models}
    chosen_names = st.multiselect("Select models to run", options=list(name_to_model.keys()), default=list(name_to_model.keys())[:1])
    st.caption(f"Tasks: {', '.join(TASKS_BY_MODALITY.get(modality, [])) or 'Custom'}")

    tabs = st.tabs(["Upload & Run", "Results Table", "Gallery", "Current Case Report"])

    with tabs[0]:
        upload = st.file_uploader("Upload images", type=["png","jpg","jpeg","bmp","tif","tiff"], accept_multiple_files=True)
        run = st.button("Run inference on selected images", type="primary")
        if run:
            if not upload:
                st.error("Please upload at least one image.")
            else:
                if not st.session_state.current_case_id:
                    meta.setdefault("modality", modality)
                    st.session_state.current_case_id = db_insert_case(meta)
                total = len(upload) * max(1, len(chosen_names))
                prog = st.progress(0); k = 0
                for uf in upload:
                    saved_image_path = save_image_file(uf)
                    content = uf.getbuffer()
                    for nm in chosen_names:
                        m = name_to_model[nm]
                        res = run_inference(m, content)
                        res["file"] = uf.name
                        res["image_path"] = saved_image_path
                        res["patient_id"] = meta.get("patient_id"); res["patient_name"] = meta.get("patient_name")
                        res["study_date"] = meta.get("study_date"); res["modality"] = modality
                        st.session_state.results.append(res)
                        try:
                            db_insert_result(st.session_state.current_case_id, res)
                        except Exception as e:
                            st.warning(f"DB save failed: {e}")
                        k += 1; prog.progress(min(1.0, k/total))
                st.success("Inference complete. See Results, Gallery, and Current Case Report tabs or the Reports page.")

    with tabs[1]:
        if st.session_state.results:
            df_local = pd.DataFrame(st.session_state.results)
            st.dataframe(df_local.tail(200), use_container_width=True)
        else:
            st.info("No results yet. Run an inference first.")

    with tabs[2]:
        rs = [r for r in st.session_state.results if r.get("overlay_b64") or r.get("boxed_b64")]
        if not rs:
            st.info("No visual overlays yet. Run a YOLO detect/segment model.")
        else:
            cols = st.columns(3)
            for i, r in enumerate(rs[-21:]):
                with cols[i % 3]:
                    b64 = r.get("overlay_b64") or r.get("boxed_b64")
                    try:
                        img_bytes = base64.b64decode(b64)
                        st.image(Image.open(io.BytesIO(img_bytes)),
                                 caption=f"{r.get('file','(image)')} — {r.get('model')}",
                                 use_container_width=True, output_format="PNG")
                    except Exception:
                        pass

    with tabs[3]:
        if not st.session_state.current_case_id:
            st.info("Create or run a case to see the live report.")
        else:
            qkey = meta.get("patient_id") or meta.get("patient_name")
            df_case = db_fetch_results({"patient_query": qkey, "modality": "All"})
            if df_case.empty:
                st.info("No stored results yet for this case.")
            else:
                st.subheader("Raw Model Outputs")
                st.dataframe(df_case, use_container_width=True)
                
                # --- Ham Rapor Özeti Oluşturma ---
                lines = [
                    f"**Patient:** {meta.get('patient_name','')} ({meta.get('patient_id','')})",
                    f"**Modality:** {modality}",
                    f"**Study Date:** {meta.get('study_date','')}",
                    "\n### AI Model Findings:",
                ]
                model_groups = df_case.groupby(['model_name','task'])['label'].apply(lambda s: ', '.join([str(x) for x in s.dropna().unique()][:5]))
                for (mname, t), labels in model_groups.items():
                    lines.append(f"- **{mname} / {t}** model found: **{labels if labels else 'N/A'}**")
                
                raw_report_text = "\n".join(lines)
                
                st.markdown("---")
                st.subheader("Draft Report Summary")
                st.markdown(raw_report_text)
                
                ### EKLENDİ: Gemini ile Rapor İyileştirme Butonu ve Mantığı ###
                st.markdown("---")
                if st.button("Raporu Gemini ile İyileştir ✨", type="primary"):
                    if not st.session_state.gemini_api_key:
                        st.error("Please enter your Gemini API key in the Settings page.")
                    else:
                        with st.spinner("Gemini is polishing the report..."):
                            polished_report = polish_with_gemini(raw_report_text, st.session_state.gemini_api_key)
                            st.session_state.polished_report_text = polished_report # Sonucu session state'e kaydet
                
                # Eğer daha önce iyileştirilmiş bir rapor varsa onu göster
                if "polished_report_text" in st.session_state and st.session_state.polished_report_text:
                    st.subheader("Gemini Polished Report")
                    st.markdown(st.session_state.polished_report_text)


def page_tabular():
    header("Tabular — Wisconsin (Diagnostic)", "Upload CSV or load built-in dataset and predict")
    st.write("**Goal**: Predict benign vs malignant (B/M). Upload CSV or load from sklearn.")
    c1, c2 = st.columns(2)
    with c1: mode = st.radio("Choose data source", ["Upload CSV", "Load sklearn sample"], index=1)
    with c2: do_train = st.checkbox("Quick-train baseline (LogReg)", value=False)

    if mode == "Upload CSV":
        csv = st.file_uploader("CSV file", type=["csv"])
        if csv is not None: df = pd.read_csv(csv)
        else: st.stop()
    else:
        try:
            from sklearn.datasets import load_breast_cancer  # type: ignore
        except Exception:
            st.error("scikit-learn not installed. Please install scikit-learn or upload a CSV."); st.stop()
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names); df["target"] = data.target

    st.dataframe(df.head(), use_container_width=True)
    target_candidates = [c for c in df.columns if c.lower() in ("diagnosis","target","label")] or [df.columns[-1]]
    target_col = st.selectbox("Target column", options=target_candidates, index=0)
    feature_cols = st.multiselect("Feature columns", options=[c for c in df.columns if c != target_col], default=[c for c in df.columns if c != target_col])

    if st.button("Run tabular prediction", type="primary"):
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
            import matplotlib.pyplot as plt
        except Exception:
            st.error("scikit-learn (and matplotlib) required."); return

        X = df[feature_cols].values; y = df[target_col].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)
        scaler = StandardScaler(); X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)
        clf = LogisticRegression(max_iter=500 if do_train else 200).fit(X_train, y_train)
        y_pred = clf.predict(X_test); y_proba = clf.predict_proba(X_test)[:,1] if hasattr(clf, "predict_proba") else None
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None and len(np.unique(y))==2 else np.nan
        st.success(f"Accuracy={acc:.3f}, ROC-AUC={auc:.3f}")
        out = pd.DataFrame({"y_true": y_test, "y_pred": y_pred}); 
        if y_proba is not None: out["y_proba"] = y_proba
        st.dataframe(out.head(30), use_container_width=True)

        cm = confusion_matrix(y_test, y_pred)
        import matplotlib.pyplot as plt
        fig = plt.figure(); plt.imshow(cm, interpolation='nearest'); plt.title('Confusion Matrix'); plt.colorbar()
        tm = np.arange(len(np.unique(y))); plt.xticks(tm, [str(t) for t in np.unique(y)], rotation=45); plt.yticks(tm, [str(t) for t in np.unique(y)])
        for i in range(cm.shape[0]): 
            for j in range(cm.shape[1]): plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center")
        plt.ylabel('True label'); plt.xlabel('Predicted label'); st.pyplot(fig)

        st.session_state.results.append({"model": "Tabular-LogReg", "modality": "Tabular", "task": "Wisconsin Predict", "ok": True,
                                         "predictions": [{"accuracy": float(acc), "auc": float(auc)}]})


def page_reports():
    header("Reports", "Select a case to generate, edit, and save a clinical report.")

    # 1. Veri ve Filtreleme
    c1, c2 = st.columns([2, 1])
    with c1:
        q = st.text_input("Search patient (ID or name)", value="")
    with c2:
        mod = st.selectbox("Modality", ["All"] + [m for m, _ in MODALITIES], index=0)
    
    st.button("Refresh List") # Sayfayı yeniden çalıştırmak için
    
    df = db_fetch_results({"patient_query": q, "modality": mod})

    if df.empty:
        st.info("No records found matching your criteria. Run inferences to create new cases.")
        return

    st.markdown("---")
    
    ### YENİ: Vaka Seçim Alanı ###
    st.subheader("1. Select a Case for Reporting")
    
    # Seçim kutusu için benzersiz vakaları hazırla
    case_options_df = df.drop_duplicates(subset=['case_id', 'patient_id', 'patient_name']).sort_values('created_at', ascending=False)
    
    # Seçenekleri kullanıcı dostu formatta oluştur
    # Örnek: "ID: 12 - Jane Doe (P001)"
    case_display_options = {
        f"ID: {row['case_id']} - {row['patient_name']} ({row['patient_id']})": row['case_id']
        for index, row in case_options_df.iterrows()
    }
    
    selected_case_display = st.selectbox(
        "Choose a case from the list below:",
        options=list(case_display_options.keys()),
        index=0,
        label_visibility="collapsed"
    )

    if not selected_case_display:
        st.stop()

    selected_case_id = case_display_options[selected_case_display]
    
    # Seçilen vakaya ait verileri filtrele
    case_df = df[df['case_id'] == selected_case_id].copy()
    case_meta = case_df.iloc[0]

    st.markdown("---")
    
    ### YENİ: Rapor Üretme ve Düzenleme Alanı ###
    st.subheader(f"2. Generate and Edit Report for Case ID: {selected_case_id}")

    # 2a. Ham Rapor Özeti
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Raw Model Findings")
        lines = [
            f"**Patient:** {case_meta.get('patient_name','')} ({case_meta.get('patient_id','')})",
            f"**Modality:** {case_meta.get('modality','')}",
            f"**Study Date:** {case_meta.get('study_date','')}",
            "\n**AI Model Findings:**",
        ]
        model_groups = case_df.groupby(['model_name','task'])['label'].apply(lambda s: ', '.join([str(x) for x in s.dropna().unique()][:5]))
        for (mname, t), labels in model_groups.items():
            lines.append(f"- **{mname} / {t}** found: **{labels if labels else 'N/A'}**")
        
        raw_report_text = "\n".join(lines)
        st.info(raw_report_text)

        # Gemini'yi tetikleme butonu
        if st.button("Generate Draft with Gemini ✨", type="primary"):
            if not st.session_state.gemini_api_key:
                st.error("Please enter your Gemini API key in the Settings page.")
            else:
                with st.spinner("Gemini is generating the report draft..."):
                    polished_report = polish_with_gemini(raw_report_text, st.session_state.gemini_api_key)
                    st.session_state[f"editable_report_{selected_case_id}"] = polished_report

    # 2b. Düzenlenebilir Nihai Rapor Alanı (İNSAN-ODAKLI)
    with col2:
        st.markdown("##### Final Report (Editable)")
        
        # Her vaka için ayrı bir session_state anahtarı kullanarak metinlerini koru
        report_key = f"editable_report_{selected_case_id}"
        if report_key not in st.session_state:
            # Varsa veritabanındaki notları yükle, yoksa boş başlat
            existing_notes = case_meta.get('notes', '')
            st.session_state[report_key] = existing_notes if existing_notes and existing_notes.strip() else "Click 'Generate Draft with Gemini' or start typing manually."

        # Düzenlenebilir metin alanı
        final_report_text = st.text_area(
            "Edit the report below. The final version can be saved to the case notes.",
            value=st.session_state[report_key],
            height=350,
            key=f"textarea_{selected_case_id}" # Her vaka için benzersiz anahtar
        )
        
        # Değişiklikleri anlık olarak session_state'e yansıt
        st.session_state[report_key] = final_report_text

        if st.button("💾 Save Final Report to Case Notes"):
            try:
                # Sadece patient_id'yi kullanarak güncelleme
                db_update_case_notes(case_meta['patient_id'], final_report_text)
                st.success(f"Report for case {selected_case_id} saved successfully to notes!")
            except Exception as e:
                st.error(f"Failed to save report: {e}")

    st.markdown("---")
    st.subheader("All Results (Raw Data Grid)")
    st.dataframe(df, use_container_width=True)
# ---------------------------
# Router & Sidebar
# ---------------------------
with st.sidebar:
    st.markdown(f"### {APP_NAME}")
    page = st.radio("Navigate", ["Home","Mammography","Ultrasound","MRI","PET","Tabular","Pathology","Models","Reports","Settings"],
                    index=["Home","Mammography","Ultrasound","MRI","PET","Tabular","Pathology","Models","Reports","Settings"].index(st.session_state.page))
    if page != st.session_state.page: st.session_state.page = page
    st.markdown("---"); st.subheader("Runtime")
    st.session_state.device = st.selectbox("Device", ["auto","cpu","cuda:0","cuda:1"], index=["auto","cpu","cuda:0","cuda:1"].index(st.session_state.device))
    st.session_state.yolo_conf = st.slider("YOLO conf", 0.0, 1.0, float(st.session_state.yolo_conf), 0.01)
    st.session_state.yolo_iou = st.slider("YOLO IoU", 0.0, 1.0, float(st.session_state.yolo_iou), 0.01)
    st.markdown("---"); st.caption("Theme: white with pink/purple accents (or your preset).")

if st.session_state.page == "Home": page_home()
elif st.session_state.page == "Models": page_models()
elif st.session_state.page == "Reports": page_reports()
elif st.session_state.page == "Settings": page_settings()
elif st.session_state.page == "Tabular": page_tabular()
else: _modality_page(st.session_state.page)
