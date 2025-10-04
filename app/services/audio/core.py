import whisper
import subprocess
import os
from typing import Optional

# ┌───────────────┬────────────┬────────────────────────┬──────────────────┐
# │ Nom du modèle │ Paramètres │ VRAM requise (approx.) │ Vitesse relative │
# ├───────────────┼────────────┼────────────────────────┼──────────────────┤
# │ tiny          │ 39 M       │ ~1 GB                  │ ~32x             │
# │ base          │ 74 M       │ ~1 GB                  │ ~16x             │
# │ small         │ 244 M      │ ~2 GB                  │ ~6x              │
# │ medium        │ 769 M      │ ~5 GB                  │ ~2x              │
# │ large         │ 1550 M     │ ~10 GB                 │ 1x               │
# └───────────────┴────────────┴────────────────────────┴──────────────────┘
MODEL_NAME = 'large-v3-turbo'
# ~/.cache/whisper/MODEL_NAME.pt
MODEL: Optional['whisper.Whisper'] = None


def get_model() -> 'whisper.Whisper':
	"""
	Loads the Whisper model if it hasn't been loaded yet, and returns the model.
	"""
	global MODEL
	if MODEL is None:
		MODEL = whisper.load_model(MODEL_NAME)
	return MODEL


def convert_webm_to_wav(webm_path: str, wav_path: str) -> None:
	subprocess.run(
		['ffmpeg', '-y', '-i', webm_path, '-ar', '16000', '-ac', '1', wav_path],
		stdout=subprocess.DEVNULL,
		stderr=subprocess.DEVNULL,
		check=True,
	)


def process_audio(webm_path: str, language: str) -> str:
	wav_path = webm_path.replace('.webm', '.wav')
	try:
		convert_webm_to_wav(webm_path, wav_path)
		result = get_model().transcribe(wav_path, language=language)
		text = result.get('text', '')
		if isinstance(text, list):
			return ' '.join(str(t) for t in text)
		return str(text)
	finally:
		if os.path.exists(wav_path):
			os.remove(wav_path)
		if os.path.exists(webm_path):
			os.remove(webm_path)
