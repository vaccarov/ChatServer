import subprocess
from functools import lru_cache

import whisper

# Model size trade-offs are tabulated in the README.
MODEL_NAME = 'large-v3-turbo'  # cached in ~/.cache/whisper/MODEL_NAME.pt


@lru_cache(maxsize=1)
def get_model() -> 'whisper.Whisper':
	"""Loads the Whisper model on first use and keeps it for the process lifetime."""
	return whisper.load_model(MODEL_NAME)


def process_audio(webm_path: str, language: str) -> str:
	"""Converts a webm recording to wav and transcribes it. Caller owns the temp directory."""
	wav_path = f'{webm_path}.wav'
	subprocess.run(
		['ffmpeg', '-y', '-i', webm_path, '-ar', '16000', '-ac', '1', wav_path],
		stdout=subprocess.DEVNULL,
		stderr=subprocess.DEVNULL,
		check=True,
	)
	return str(get_model().transcribe(wav_path, language=language).get('text', ''))
