from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import tempfile
import os
import gc
import re
import difflib
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
    Whisper のセグメント単位のタイミングをアンカーとして、
    ネット取得の正確な歌詞テキストを配置する。

    word-level matching ではなく segment-level matching を使うことで、
    Whisper の聞き違いによるタイミングのズレを防ぐ。
    """
    if not whisper_segments:
        return whisper_segments

    # 歌詞をパース（空行・セクションマーカーを除去）
    raw_lines = fetched_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    fetched_lines = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        # [Chorus] 【サビ】 (繰り返し) などのメタ行を除去
        if re.match(r'^[\[【（(].+[\]】）)]$', line):
            continue
        fetched_lines.append(line)

    if not fetched_lines:
        return whisper_segments

    def norm(s: str) -> str:
        """句読点除去・小文字化（英語・日本語両対応）"""
        return re.sub(r'[^\w\s]', '', s.lower(), flags=re.UNICODE).strip()

    w_norm = [norm(seg['text']) for seg in whisper_segments]
    f_norm = [norm(line) for line in fetched_lines]

    # セグメント単位でシーケンスマッチング
    matcher = difflib.SequenceMatcher(None, w_norm, f_norm, autojunk=False)
    matched: dict = {}  # fetched_line_idx -> (start, end)

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == 'equal':
            for d in range(i2 - i1):
                wseg = whisper_segments[i1 + d]
                matched[j1 + d] = (wseg['start'], wseg['end'])

    total = len(fetched_lines)
    n_w   = len(whisper_segments)
    match_ratio = len(matched) / max(n_w, total, 1)

    if match_ratio < 0.15:
        # テキスト類似度が低い場合（言語違い等）→ 位置比率でセグメントを割り当て
        logger.info("アライメント: 比率マッピングに切替 (match_ratio=%.2f)", match_ratio)
        matched = {}
        groups: dict = {}
        for i in range(total):
            w_idx = min(int(i * n_w / total), n_w - 1)
            groups.setdefault(w_idx, []).append(i)
        for w_idx, fi_list in groups.items():
            wseg = whisper_segments[w_idx]
            n = len(fi_list)
            seg_dur = (wseg['end'] - wseg['start']) / n if n else 0
            for k, fi in enumerate(fi_list):
                t = wseg['start'] + k * seg_dur
                matched[fi] = (round(t, 3), round(t + seg_dur, 3))

    song_start = whisper_segments[0]['start']
    song_end   = whisper_segments[-1]['end']
    known = sorted(matched)

    if not known:
        # フォールバック: 均等分配
        dur = (song_end - song_start) / total if total else 1.0
        for i in range(total):
            t = song_start + i * dur
            matched[i] = (round(t, 3), round(t + dur, 3))
        known = list(range(total))

    # 未マッチ行を前後アンカーから補間
    fa = known[0]
    if fa > 0:
        t0  = matched[fa][0]
        dur = min(t0 / fa if fa else 0.5, 3.0)
        for i in range(fa):
            t = max(0.0, t0 - (fa - i) * dur)
            matched[i] = (round(t, 3), round(t + dur, 3))

    for a, b in zip(known, known[1:]):
        if b - a <= 1:
            continue
        t_a_end   = matched[a][1]
        t_b_start = matched[b][0]
        gap  = b - a
        span = max(0.0, t_b_start - t_a_end)
        dur  = span / gap if gap else 0.0
        for i in range(a + 1, b):
            frac = (i - a) / gap
            t    = t_a_end + frac * span
            matched[i] = (round(t, 3), round(min(t + dur, t_b_start), 3))

    la = known[-1]
    if la < total - 1:
        _, t_end  = matched[la]
        remaining = total - 1 - la
        span = max(song_end - t_end, remaining * 0.5)
        dur  = span / remaining if remaining else 0.5
        for i in range(la + 1, total):
            t = t_end + (i - la) * dur
            matched[i] = (round(t, 3), round(t + dur, 3))

    # 各行のセグメントを構築（単語はセグメント時間内で均等分配）
    result = []
    for i, line in enumerate(fetched_lines):
        s, e = matched.get(i, (song_start, song_end))
        words    = line.split()
        n        = len(words)
        word_dur = (e - s) / n if n else (e - s)
        seg_words = [
            {
                'start': round(s + wi * word_dur, 3),
                'end':   round(s + (wi + 1) * word_dur, 3),
                'word':  (' ' + w) if wi > 0 else w,
            }
            for wi, w in enumerate(words)
        ]
        result.append({
            'start': round(s, 3),
            'end':   round(e, 3),
            'text':  line,
            'words': seg_words,
        })

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
