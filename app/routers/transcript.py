import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.whisper_utils import process_audio

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@router.post("/decode")
async def transcribe(file: UploadFile = File(...), language: str = Form(...)) -> dict[str, Any]:
    webm = UPLOAD_DIR / f"{uuid.uuid4().hex}.webm"

    with open(webm, "wb") as f:
        f.write(await file.read())
    try:
        transcript = await run_in_threadpool(process_audio, webm, language)
        return {"transcript": transcript}
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        raise HTTPException(
            status_code=500, detail="An error occurred during the transcription process."
        )
