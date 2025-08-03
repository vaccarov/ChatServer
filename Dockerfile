FROM python:3.9-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /root/.cache/whisper && python -c "import whisper; whisper.load_model('large-v3-turbo', download_root='/root/.cache/whisper')"

FROM python:3.9-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg
WORKDIR /app
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /root/.cache/whisper /root/.cache/whisper
COPY ./app /app/app
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]