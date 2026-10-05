import io
from functools import lru_cache

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()


class TtsRequest(BaseModel):
	text: str


@lru_cache(maxsize=1)
def _get_model():
	"""Loads VoxCPM on first use so the server can start without downloading it."""
	from voxcpm import VoxCPM

	return VoxCPM.from_pretrained('openbmb/VoxCPM-0.5B')


def _synthesize(text: str) -> io.BytesIO:
	model = _get_model()
	wav = model.generate(
		text=text,
		prompt_wav_path=None,  # optional: path to a prompt speech for voice cloning
		prompt_text=None,  # optional: reference text
		cfg_value=2.0,  # LM guidance on LocDiT; higher adheres more to the prompt
		inference_timesteps=10,  # higher for better quality, lower for speed
		normalize=True,  # enable external TN tool
		denoise=True,  # enable external Denoise tool
		retry_badcase=True,  # retry mode for some bad cases (unstoppable)
		retry_badcase_max_times=3,
		retry_badcase_ratio_threshold=6.0,  # max length for bad case detection
	)
	buffer = io.BytesIO()
	model.save(wav, buffer)
	buffer.seek(0)
	return buffer


@router.post('')
async def tts(req: TtsRequest) -> StreamingResponse:
	return StreamingResponse(await run_in_threadpool(_synthesize, req.text), media_type='audio/wav')
