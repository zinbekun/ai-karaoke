from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import tempfile
import os
import gc
import traceback
import logging
import asyncio

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

def convert_to_wav(src: str) -> str:
    """ffmpegで任意フォーマットをWAVに変換"""
    import shutil, subprocess
    dst = src + "_converted.wav"
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run(
        [ffmpeg, "-y", "-i", src,
         "-ar", "22050", "-ac", "1", "-f", "wav", dst],
        check=True, capture_output=True, timeout=120
    )
    return dst


def do_pitch(audio_path: str) -> dict:
    """音程解析 (pYIN) — 完了後に大きなarrayをfreeする"""
    ext = os.path.splitext(audio_path)[1].lower()
    wav_path = None
    if ext not in {".wav", ".flac", ".ogg"}:
        logger.info("WAVに変換中: %s", ext)
        wav_path = convert_to_wav(audio_path)
        load_path = wav_path
    else:
        load_path = audio_path

    try:
        y, sr = librosa.load(load_path, sr=22050, mono=True, duration=MAX_DURATION)
    finally:
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)

    duration = float(librosa.get_duration(y=y, sr=sr))
    hop_length = 512

    # HPSS: 打楽器成分を除去し、ボーカルのような持続的な倍音成分を抽出する
    # margin が高いほど分離が積極的になる（8 は強めの分離）
    logger.info("HPSS でボーカル成分を分離中...")
    y_harmonic = librosa.effects.harmonic(y, margin=8)
    del y  # 元波形はもう不要

    # 倍音成分だけで pYIN を実行 → ボーカルの音程を追跡
    # fmin/fmax を人声の実用域（E2〜C6）に絞ることで楽器誤検出を減らす
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y_harmonic,
        fmin=librosa.note_to_hz("E2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
        hop_length=hop_length,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    def safe_midi(freq):
        return max(0.0, min(127.0, float(librosa.hz_to_midi(freq))))

    def midi_note(m):
        return librosa.midi_to_note(int(round(max(0.0, min(127.0, m)))))

    # voiced_probs > 0.5 かつ voiced_flag でフィルタ（pyin のデフォルトより少し厳しめ）
    raw_pitch = [
        {"t": round(float(t), 3), "midi": round(safe_midi(f), 2)}
        for t, f, v, p in zip(times, f0, voiced_flag, voiced_probs)
        if v and not np.isnan(f) and p > 0.5
    ]

    voiced_frames = [
        (i, float(times[i]), safe_midi(f0[i]))
        for i in range(len(times))
        if voiced_flag[i] and not np.isnan(f0[i]) and voiced_probs[i] > 0.5
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

    result = {"duration": duration, "segments": segments, "raw_pitch": raw_pitch}

    # 大きなarrayを明示的に解放してRAMを節約
    del y_harmonic, f0, voiced_flag, voiced_probs, times, voiced_frames
    gc.collect()

    return result


async def identify_song(audio_path: str) -> tuple:
    """Shazamで曲名・アーティスト名を認識"""
    try:
        from shazamio import Shazam
        shazam = Shazam()
        out = await asyncio.wait_for(shazam.recognize(audio_path), timeout=20)
        if out and "track" in out:
            track  = out["track"]
            title  = track.get("title", "")
            artist = track.get("subtitle", "")
            if title:
                return title, artist
    except asyncio.TimeoutError:
        logger.warning("Shazam認識タイムアウト")
    except Exception as e:
        logger.warning("Shazam認識エラー: %s", e)
    return None, None







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

        # Step 1: 音程解析（内部でgc.collect済み）
        logger.info("音程解析中...")
        pitch_data = do_pitch(tmp_path)

        # Step 2: 曲名認識（表示用）
        title, artist = None, None
        try:
            title, artist = await identify_song(tmp_path)
            if title:
                logger.info("曲名認識: %s - %s", artist, title)
            else:
                logger.info("曲名認識失敗")
        except Exception as e:
            logger.warning("曲名取得エラー: %s", e)

        logger.info("解析完了: duration=%.1fs, pitch_segs=%d",
                    pitch_data["duration"], len(pitch_data["segments"]))

        return {
            "filename":  file.filename,
            "duration":  pitch_data["duration"],
            "segments":  pitch_data["segments"],
            "raw_pitch": pitch_data["raw_pitch"],
            "song_info": {"title": title, "artist": artist} if title else None,
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
