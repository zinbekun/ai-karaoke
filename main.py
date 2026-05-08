from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import tempfile
import os
import gc
import re
import traceback
import logging
import threading
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

    result = {"duration": duration, "segments": segments, "raw_pitch": raw_pitch}

    # 大きなarrayを明示的に解放してRAMを節約
    del y, f0, voiced_flag, times, voiced_frames
    gc.collect()

    return result


def do_transcribe(audio_path: str) -> list:
    """Whisperで歌詞を文字起こし"""
    model = get_whisper_model()
    segs, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        beam_size=3,
        vad_filter=True,
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


async def fetch_lyrics_ovh(artist: str, title: str):
    """LyricsOVH APIから歌詞取得（主に英語）"""
    try:
        import httpx
        from urllib.parse import quote
        url = f"https://api.lyrics.ovh/v1/{quote(artist)}/{quote(title)}"
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            if r.status_code == 200:
                text = r.json().get("lyrics", "").strip()
                if text:
                    return text
    except Exception as e:
        logger.debug("LyricsOVH失敗: %s", e)
    return None


async def fetch_lyrics_genius(artist: str, title: str):
    """Genius公開検索から歌詞取得（日本語を含む多言語）"""
    try:
        import httpx, html as html_mod
        query = f"{artist} {title}"
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
        }
        search_url = "https://genius.com/api/search/song"
        async with httpx.AsyncClient(timeout=15, headers=headers, follow_redirects=True) as client:
            sr = await client.get(search_url, params={"q": query})
            if sr.status_code != 200:
                return None
            sections = sr.json().get("response", {}).get("sections", [])
            hits = sections[0].get("hits", []) if sections else []
            if not hits:
                return None
            song_url = hits[0].get("result", {}).get("url")
            if not song_url:
                return None

            lr = await client.get(song_url)
            if lr.status_code != 200:
                return None

            containers = re.findall(
                r'<div data-lyrics-container="true"[^>]*>(.*?)</div>',
                lr.text, re.DOTALL
            )
            if not containers:
                return None

            parts = []
            for c in containers:
                c = re.sub(r'<br\s*/?>', '\n', c, flags=re.IGNORECASE)
                c = re.sub(r'<[^>]+>', '', c)
                c = html_mod.unescape(c)
                parts.append(c.strip())

            lyrics = '\n'.join(parts).strip()
            return lyrics if lyrics else None
    except Exception as e:
        logger.debug("Genius取得失敗: %s", e)
    return None


async def fetch_lyrics(artist: str, title: str):
    """LyricsOVH → Genius の順で歌詞取得"""
    lyrics = await fetch_lyrics_ovh(artist, title)
    if lyrics:
        logger.info("歌詞取得成功 (LyricsOVH): %d文字", len(lyrics))
        return lyrics, "LyricsOVH"
    lyrics = await fetch_lyrics_genius(artist, title)
    if lyrics:
        logger.info("歌詞取得成功 (Genius): %d文字", len(lyrics))
        return lyrics, "Genius"
    return None, None


def align_lyrics(fetched_text: str, whisper_segments: list) -> list:
    """
    Whisper の単語タイムスタンプを「位置比率」でフェッチ歌詞の各単語に割り当てる。

    テキストマッチングに依存しないため、Whisper の聞き違いや言語の違いに
    関係なく安定して動作する。

    原理: Whisper が N 単語を出力し、歌詞が M 単語あるとき、
          歌詞の j 番目の単語は Whisper の (j * N / M) 番目の単語と
          ほぼ同じ時刻に歌われていると仮定する。
    """
    if not whisper_segments:
        return whisper_segments

    # フェッチ歌詞のパース（空行・セクションマーカーを除去）
    raw_lines = fetched_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    fetched_lines = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r'^[\[【（(].+[\]】）)]$', line):  # [Chorus] 【サビ】 等
            continue
        fetched_lines.append(line)

    if not fetched_lines:
        return whisper_segments

    # Whisper 単語タイムスタンプを収集
    ww = []  # [{start, end}, ...]
    for seg in whisper_segments:
        for w in seg.get('words', []):
            if w.get('word', '').strip():
                ww.append({'start': w['start'], 'end': w['end']})

    # word timestamps がない場合: セグメントを1点として使う
    if not ww:
        for seg in whisper_segments:
            ww.append({'start': seg['start'], 'end': seg['end']})

    if not ww:
        return whisper_segments

    # フェッチ歌詞の全単語をフラット化: (word, line_idx, word_idx_in_line)
    flat = []
    for li, line in enumerate(fetched_lines):
        for wi, word in enumerate(line.split()):
            flat.append((word, li, wi))

    if not flat:
        return whisper_segments

    n_ww = len(ww)
    n_fw = len(flat)

    # 各フェッチ単語に Whisper タイムスタンプを線形補間で割り当て
    from collections import defaultdict
    by_line = defaultdict(list)

    for j, (word, li, wi) in enumerate(flat):
        # 位置比率: 0.0 〜 n_ww-1 の実数インデックス
        pos    = j * (n_ww - 1) / (n_fw - 1) if n_fw > 1 else 0.0
        lo     = int(pos)
        hi     = min(lo + 1, n_ww - 1)
        frac   = pos - lo
        ts = ww[lo]['start'] + frac * (ww[hi]['start'] - ww[lo]['start'])
        te = ww[lo]['end']   + frac * (ww[hi]['end']   - ww[lo]['end'])
        by_line[li].append({
            'word':  (' ' + word) if wi > 0 else word,
            'start': round(ts, 3),
            'end':   round(te, 3),
        })

    # ライン単位でセグメントを構築
    result = []
    for li, line in enumerate(fetched_lines):
        words_in_line = by_line.get(li, [])
        if not words_in_line:
            continue
        result.append({
            'start': words_in_line[0]['start'],
            'end':   words_in_line[-1]['end'],
            'text':  line,
            'words': words_in_line,
        })

    logger.info("アライメント完了: whisper_words=%d, lyrics_words=%d, lines=%d",
                n_ww, n_fw, len(result))
    return result


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

        # Step 2: 曲名認識 & 歌詞取得（軽量・非同期）
        title, artist, fetched_lyrics, lyrics_source = None, None, None, None
        try:
            title, artist = await identify_song(tmp_path)
            if title:
                logger.info("曲名認識: %s - %s", artist, title)
                fetched_lyrics, lyrics_source = await fetch_lyrics(artist or "", title)
            else:
                logger.info("曲名認識失敗（Whisper歌詞のみ使用）")
        except Exception as e:
            logger.warning("曲名/歌詞取得エラー: %s", e)

        # Step 3: Whisper文字起こし
        logger.info("歌詞文字起こし中...")
        try:
            lyrics = do_transcribe(tmp_path)
            logger.info("歌詞セグメント数: %d", len(lyrics))
        except Exception as e:
            logger.warning("歌詞文字起こし失敗 (スキップ): %s", e)
            lyrics = []
        gc.collect()

        # Step 4: 歌詞アライメント（ネット歌詞 + Whisperタイミング）
        if fetched_lyrics and lyrics:
            try:
                lyrics = align_lyrics(fetched_lyrics, lyrics)
                logger.info("アライメント完了: %dセグメント (source=%s)", len(lyrics), lyrics_source)
            except Exception as e:
                logger.warning("アライメント失敗（Whisper歌詞を使用）: %s", e)
                lyrics_source = None

        logger.info("解析完了: duration=%.1fs, pitch_segs=%d, lyrics=%d",
                    pitch_data["duration"], len(pitch_data["segments"]), len(lyrics))

        return {
            "filename":     file.filename,
            "duration":     pitch_data["duration"],
            "segments":     pitch_data["segments"],
            "raw_pitch":    pitch_data["raw_pitch"],
            "lyrics":       lyrics,
            "song_info":    {"title": title, "artist": artist, "source": lyrics_source} if title else None,
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
