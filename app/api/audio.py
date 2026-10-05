import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.services.audio.core import process_audio

router = APIRouter()


@router.post('/decode')
async def transcribe(file: UploadFile = File(...), language: str = Form(...)) -> dict[str, Any]:
	with tempfile.TemporaryDirectory() as temp_dir:
		webm = Path(temp_dir) / 'audio.webm'
		webm.write_bytes(await file.read())
		try:
			transcript = await run_in_threadpool(process_audio, str(webm), language)
		except Exception as e:
			print(f'An error occurred during transcription: {e}')
			raise HTTPException(status_code=500, detail='An error occurred during the transcription process.')
	return {'transcript': transcript}
