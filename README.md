# Deepfake Authentication Agent

**Live:**  
- **App (UI):** https://deepfake-auth-agent.onrender.com/app/  
- **API Docs:** https://deepfake-auth-agent.onrender.com/docs  
- **Health:** https://deepfake-auth-agent.onrender.com/health

Detect likely deepfakes in images using two methods:
- **ELA (baseline):** error-level analysis via JPEG recompression residuals.
- **Deep (EfficientNet-B0):** trained PyTorch classifier (timm) on frames extracted from real/fake datasets.

> For research/education only. Not medical, legal, or forensic advice.

---

## Features

- FastAPI backend with OpenAPI/Swagger at `/docs`
- Single-page frontend (Tailwind) served at `/app/`
- Drag-and-drop image upload, preview, spinner, verdict badge, progress bar, JSON details, mini history
- Two detectors: `ela` (no ML), `deep` (EfficientNet-B0)
- Training script & dataset frame extractor (supports videos and image-sequence folders, even extension-less frames)

---

## Quickstart (local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
# for deep model locally:
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install timm
uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
# UI:   http://127.0.0.1:8000/app/
# Docs: http://127.0.0.1:8000/docs


