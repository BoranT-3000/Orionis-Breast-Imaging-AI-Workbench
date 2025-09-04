# Orionis - Breast-Imaging-AI-Workbench
A Streamlit-based AI workbench for medical imaging with a focus on breast imaging. Upload your own models and run object detection, segmentation, and classification on images. Also includes a tabular module (e.g., Wisconsin Diagnostic dataset) for quick ML baselines.

⚠️ Medical Disclaimer: This is a proof-of-concept research tool. Not a medical device. Do not use for clinical decision-making.


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
