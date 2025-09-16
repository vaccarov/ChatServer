from app.whisper_utils import process_audio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
import uuid, os
from typing import Any

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def health_check():
    return {"success": True}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form(...)) -> dict[str, Any]:
    webm = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.webm")

    with open(webm, "wb") as f:
        f.write(await file.read())
    try:
        transcript = await run_in_threadpool(process_audio, webm, language)
        return {"transcript": transcript}
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during the transcription process.")
