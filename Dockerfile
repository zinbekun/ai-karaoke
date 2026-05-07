FROM python:3.11-slim

# ffmpeg（M4A/MP3/AACなど全フォーマット対応）
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Whisperモデルをビルド時にダウンロード（起動を速くするため）
ARG WHISPER_MODEL=base
RUN python -c "\
from faster_whisper import WhisperModel; \
print('Whisperモデルをダウンロード中...'); \
WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8'); \
print('完了')"

COPY . .

ENV WHISPER_MODEL=base
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
