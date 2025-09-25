from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pathlib import Path
from pydantic import BaseModel
from typing import Optional
from models.image_ela import analyze_image_bytes as analyze_image_ela 
from models.image_deep import analyze_image_bytes as analyze_image_deep
from models.video_frames import analyze_video_bytes
from models.audio_spectrogram import analyze_audio_bytes

app = FastAPI(title="Deepfake Detection & Media Authentication Agent")

FRONTEND_DIR = (Path(__file__).resolve().parent.parent / "frontend").resolve()
app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="app")

@app.get("/")
def root():
    return RedirectResponse("/app/")



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Health(BaseModel):
    status: str = "ok"

@app.get("/health", response_model=Health)
def health():
    return Health()

@app.post("/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
    method: str = Form("ela")
):
    data = await file.read()
    choice = (method or "ela").lower()

    if choice == "deep":
        try:
            result = analyze_image_deep(data)
        except Exception as e:
            result, _ = analyze_image_ela(data)
            result["warning"] = f"deep model unavailable: {type(e).__name__}: {e}"
    else:
        result, _ = analyze_image_ela(data)

    return JSONResponse(result)


@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...), fps: Optional[int] = Form(2)):
    data = await file.read()
    result, _ = analyze_video_bytes(data, fps=fps or 2)
    return JSONResponse(result)

@app.post("/analyze/audio")
async def analyze_audio(file: UploadFile = File(...)):
    data = await file.read()
    result = analyze_audio_bytes(data)
    return JSONResponse(result)
