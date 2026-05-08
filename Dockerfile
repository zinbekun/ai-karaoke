FROM python:3.11-slim

# システムパッケージ
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pythonパッケージ（段階的にインストールしてメモリ節約）
COPY requirements.txt .
RUN pip install --no-cache-dir numpy scipy && \
    pip install --no-cache-dir librosa soundfile && \
    pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart && \
    pip install --no-cache-dir faster-whisper && \
    pip install --no-cache-dir shazamio httpx

COPY . .

# モデルキャッシュの保存先
ENV HF_HOME=/app/.cache
ENV WHISPER_MODEL=tiny

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
