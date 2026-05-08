from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import tempfile
import os
import traceback
import logging
import threading
import concurrent.futures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("server.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AutoKaraoke")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_DURATION  = 360  # 6 minutes

# Whisper model — loaded once on first use
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")

_whisper_model = None
_whisper_lock  = threading.Lock()


def get_whisper_model():
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            logger.info("Whisperモデルをロード中 [%s] (初回のみ)...", WHISPER_MODEL)
            from faster_whisper import WhisperModel
            _whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
            logger.info("Whisperモデル ロード完了")
    return _whisper_model


def do_pitch(audio_path: str) -> dict:
    """音程解析 (pYIN)"""
    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=MAX_DURATION)
    duration = float(librosa.get_duration(y=y, sr=sr))
    hop_length = 512

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=hop_length,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    def safe_midi(freq):
        return max(0.0, min(127.0, float(librosa.hz_to_midi(freq))))

    def midi_note(m):
        return librosa.midi_to_note(int(round(max(0.0, min(127.0, m)))))

    raw_pitch = [
        {"t": round(float(t), 3), "midi": round(safe_midi(f), 2)}
        for t, f, v in zip(times, f0, voiced_flag)
        if v and not np.isnan(f)
    ]

    voiced_frames = [
        (i, float(times[i]), safe_midi(f0[i]))
        for i in range(len(times))
        if voiced_flag[i] and not np.isnan(f0[i])
    ]

    segments = []
    if voiced_frames:
        group = [voiced_frames[0]]
        for frame in voiced_frames[1:]:
            if frame[0] - group[-1][0] <= 3:
                group.append(frame)
            else:
                if len(group) >= 3:
                    midis = [f[2] for f in group]
                    m = float(np.median(midis))
                    segments.append({
                        "start": round(group[0][1], 3),
                        "end":   round(group[-1][1] + hop_length / sr, 3),
                        "midi":  round(m, 1),
                        "note":  midi_note(m),
                    })
                group = [frame]
        if len(group) >= 3:
            midis = [f[2] for f in group]
            m = float(np.median(midis))
            segments.append({
                "start": round(group[0][1], 3),
                "end":   round(group[-1][1] + hop_length / sr, 3),
                "midi":  round(m, 1),
                "note":  midi_note(m),
            })

    return {"duration": duration, "segments": segments, "raw_pitch": raw_pitch}


def do_transcribe(audio_path: str) -> list:
    """Whisperで歌詞を文字起こし"""
    model = get_whisper_model()
    segs, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        beam_size=3,
        vad_filter=True,           # 無音区間をスキップ
        vad_parameters={"min_silence_duration_ms": 500},
    )
    logger.info("Whisper言語検出: %s", info.language)

    lyrics = []
    for seg in segs:
        words = []
        if seg.words:
            words = [
                {"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word}
                for w in seg.words
            ]
        lyrics.append({
            "start": round(seg.start, 3),
            "end":   round(seg.end, 3),
            "text":  seg.text.strip(),
            "words": words,
        })
    return lyrics


@app.post("/api/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    allowed = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"未対応のフォーマットです: {ext}")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, "ファイルが大きすぎます (最大 100MB)")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        logger.info("解析開始: %s", file.filename)

        # 音程解析と歌詞文字起こしを並列実行
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            pitch_future  = pool.submit(do_pitch, tmp_path)
            lyrics_future = pool.submit(do_transcribe, tmp_path)

            pitch_data = pitch_future.result()

            try:
                lyrics = lyrics_future.result()
                logger.info("歌詞セグメント数: %d", len(lyrics))
            except Exception as e:
                logger.warning("歌詞文字起こし失敗 (スキップ): %s", e)
                lyrics = []

        logger.info("解析完了: duration=%.1fs, segments=%d, lyrics=%d",
                    pitch_data["duration"], len(pitch_data["segments"]), len(lyrics))

        return {
            "filename": file.filename,
            "duration": pitch_data["duration"],
            "segments": pitch_data["segments"],
            "raw_pitch": pitch_data["raw_pitch"],
            "lyrics":   lyrics,
        }

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("解析失敗:\n%s", tb)
        raise HTTPException(500, f"解析エラー: {str(e)}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.get("/api/health")
async def health():
    import subprocess, shutil
    ffmpeg_path = shutil.which("ffmpeg")
    ffmpeg_ok = False
    if ffmpeg_path:
        try:
            r = subprocess.run([ffmpeg_path, "-version"], capture_output=True, timeout=5)
            ffmpeg_ok = r.returncode == 0
        except Exception:
            pass
    return {
        "status": "ok",
        "ffmpeg_path": ffmpeg_path,
        "ffmpeg_ok": ffmpeg_ok,
        "whisper_model": WHISPER_MODEL,
    }

@app.get("/api/logs")
async def get_logs():
    try:
        with open("server.log", encoding="utf-8") as f:
            lines = f.readlines()
        return {"logs": "".join(lines[-80:])}
    except FileNotFoundError:
        return {"logs": "(ログファイルなし)"}

app.mount("/", StaticFiles(directory="static", html=True), name="static")
