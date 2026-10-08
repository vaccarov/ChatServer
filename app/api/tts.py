import io
import threading
from functools import lru_cache

import soundfile
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()
_synthesis_lock = threading.Lock()


class TtsRequest(BaseModel):
	text: str


@lru_cache(maxsize=1)
def _get_model():
	"""Loads VoxCPM on first use so the server can start without downloading it."""
	from voxcpm import VoxCPM

	return VoxCPM.from_pretrained('openbmb/VoxCPM2')


def _synthesize(text: str) -> io.BytesIO:
	if not any(char.isalnum() for char in text):
		raise HTTPException(status_code=400, detail='Nothing to synthesize: the text contains no words.')

	with _synthesis_lock:
		model = _get_model()
		wav = model.generate(
			text=text,
			prompt_wav_path=None,  # optional: reference audio for voice cloning
			prompt_text=None,  # optional: reference transcript
			cfg_value=2.0,  # LM guidance on LocDiT; higher adheres more to the prompt
			inference_timesteps=10,  # higher for better quality, lower for speed
			normalize=True,  # enable external TN tool
			denoise=True,  # enable external Denoise tool
			retry_badcase=True,  # retry mode for some bad cases (unstoppable)
			retry_badcase_max_times=3,
			retry_badcase_ratio_threshold=6.0,  # max length for bad case detection
		)
	buffer = io.BytesIO()
	soundfile.write(buffer, wav, model.tts_model.sample_rate, format='WAV')
	buffer.seek(0)
	return buffer


@router.post('')
async def tts(req: TtsRequest) -> StreamingResponse:
	return StreamingResponse(await run_in_threadpool(_synthesize, req.text), media_type='audio/wav')
