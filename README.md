# Orionis - Breast-Imaging-AI-Workbench
> A Streamlit-based AI workbench for medical imaging with a focus on breast imaging. Upload your own models and run object detection, segmentation, and classification on images. Also includes a tabular module (e.g., Wisconsin Diagnostic dataset) for quick ML baselines.

⚠️ **Medical Disclaimer:** This is a proof-of-concept research tool. Not a medical device. Do not use for clinical decision-making.

---

## Key Features

- **Model Registry:** Add YOLO / Keras / PyTorch / ONNX models with metadata.
Inference:
  - YOLO (detect/segment) with optional overlays/masks.
  - Keras / TensorFlow & PyTorch classification.
  - ONNX inference (CPU by default).

- **Cases & Results:**
  - Store patient/study metadata and results in SQLite.
  - Results table, image gallery (overlays), and case reports.

- **Reporting:**
  - Auto-generate a draft report from model outputs.
  - Optional Gemini polish for clinician-friendly wording.

- **Tabular Module:**
  - Wisconsin Diagnostic demo with quick Logistic Regression baseline.

- **Theming & Runtime:**
  - Theme colors, compact density.

- **Device selector:** auto | cpu | cuda:0 | cuda:1.

---

## **Directory Layout (auto-created on first run)**

breast_ai_workbench/
├─ db/ # SQLite DB
├─ img/ # Uploaded images
└─ models/ # Uploaded model files

---

## 🧱 Tech Stack
- UI: **Streamlit**
- Data: **SQLite**, **pandas**
- Imaging: **Pillow (PIL)**
- AI Backends (plug-and-play / optional)
  - **ultralytics (YOLO)** + (optional) **torch**
  - **tensorflow/keras**
  - **onnxruntime**
  - **scikit-learn** (tabular)
- Optional polishing: **google-generativeai (Gemini)**

---

## Quickstart

### 1) Environment
- **Python** 3.10–3.12 recommended
- (Optional) **CUDA** + compatible **PyTorch** if you want GPU

```bash
# clone
git clone https://github.com/BoranT-3000/Orionis---Breast-Imaging-AI-Workbench
cd Orionis---Breast-Imaging-AI-Workbench

# venv
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# install
pip install --upgrade pip
pip install -r requirements.txt

# run
streamlit run app.py

