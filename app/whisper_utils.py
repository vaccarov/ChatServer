import whisper
import subprocess
import os

# ~/.cache/whisper/__model__.pt
MODEL = whisper.load_model("large-v3-turbo")

def convert_webm_to_wav(webm_path: str, wav_path: str) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-i", webm_path, "-ar", "16000", "-ac", "1", wav_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def process_audio(webm_path: str, language: str) -> str:
    wav_path = webm_path.replace(".webm", ".wav")
    try:
        convert_webm_to_wav(webm_path, wav_path)
        result = MODEL.transcribe(wav_path, language=language)
        return result.get("text", "")
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        if os.path.exists(webm_path):
            os.remove(webm_path)
