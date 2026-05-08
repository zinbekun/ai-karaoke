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
import unicodedata
import traceback
from collections import defaultdict
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




def _merge_segs(segs: list, gap: float = 1.5) -> list:
    """Whisper セグメントの小さなギャップを結合してボーカル区間リストを返す"""
    if not segs:
        return []
    out = [{'start': segs[0]['start'], 'end': segs[0]['end']}]
    for s in segs[1:]:
        if s['start'] - out[-1]['end'] <= gap:
            out[-1]['end'] = max(out[-1]['end'], s['end'])
        else:
            out.append({'start': s['start'], 'end': s['end']})
    return out


def _vt_to_abs(vt: float, voiced: list) -> float:
    """ボーカル累積時間 → 絶対時間（ギャップをスキップして戻す）"""
    rem = vt
    for seg in voiced:
        d = seg['end'] - seg['start']
        if rem <= d:
            return seg['start'] + rem
        rem -= d
    return (voiced[-1]['end'] + rem) if voiced else vt


def _abs_to_vt(t: float, voiced: list) -> float:
    """絶対時間 → ボーカル累積時間（ギャップの時間を除外）"""
    vt = 0.0
    for seg in voiced:
        if t <= seg['start']:
            break
        if t >= seg['end']:
            vt += seg['end'] - seg['start']
        else:
            vt += t - seg['start']
    return vt


def _uniform_segments(lines: list, t0: float, t1: float) -> list:
    """均等時間分配フォールバック（ボーカル情報なし時）"""
    n    = len(lines)
    span = max(t1 - t0, 1.0)
    out  = []
    for i, line in enumerate(lines):
        ts       = t0 + i       * span / n
        te       = t0 + (i + 1) * span / n
        words    = line.split()
        nw       = len(words)
        wd       = (te - ts) / nw if nw else (te - ts)
        out.append({
            'start': round(ts, 3),
            'end':   round(te, 3),
            'text':  line,
            'words': [
                {'start': round(ts + wi * wd, 3),
                 'end':   round(ts + (wi+1) * wd, 3),
                 'word':  (' ' + w) if wi > 0 else w}
                for wi, w in enumerate(words)
            ],
        })
    return out




def align_lyrics(fetched_text: str, whisper_segments: list, audio_duration: float = 0.0) -> list:
    """
    Whisper セグメントとネット歌詞行を類似度マッチング後、
    未マッチ行を「曲の平均テンポ」で補間する。

    - アンカー行（Whisper マッチ済み）は絶対に時刻を変更しない
    - MIN_GAP カスケードはアンカーで遮断 → 「止まり→高速」バグを防ぐ
    - 時間が足りないゾーンは超過行をスキップし次アンカーへ正確にジャンプ
    """
    if not whisper_segments:
        return whisper_segments

    fetched_lines = []
    for line in fetched_text.replace('\r\n', '\n').replace('\r', '\n').split('\n'):
        line = line.strip()
        if not line or re.match(r'^[\[【（(].+[\]】）)]$', line):
            continue
        fetched_lines.append(line)
    if not fetched_lines:
        return whisper_segments

    song_start  = whisper_segments[0]['start']
    whisper_end = whisper_segments[-1]['end']
    if audio_duration > 0 and whisper_end < audio_duration * 0.80:
        song_end = audio_duration * 0.88
        logger.info("アライメント: 終端補正 %.1f→%.1f秒", whisper_end, song_end)
    else:
        song_end = whisper_end

    voiced      = _merge_segs(whisper_segments, gap=2.0)
    total_vocal = sum(s['end'] - s['start'] for s in voiced)
    n_lines     = len(fetched_lines)

    if total_vocal <= 0:
        return _uniform_segments(fetched_lines, song_start, song_end)

    # 曲の平均テンポ（1行あたりのボーカル累積時間）
    ideal_vt = total_vocal / n_lines

    def norm_text(s: str) -> str:
        return re.sub(r'[^\w]', '', unicodedata.normalize('NFKC', s), flags=re.UNICODE).lower()

    def make_words(line: str, t_s: float, t_e: float) -> list:
        words = line.split()
        nw = len(words)
        wd = (t_e - t_s) / nw if nw > 0 else max(t_e - t_s, 0.3)
        return [
            {'start': round(t_s + wi * wd, 3),
             'end':   round(t_s + (wi + 1) * wd, 3),
             'word':  (' ' + w) if wi > 0 else w}
            for wi, w in enumerate(words)
        ]

    # ── セグメントレベル類似度マッチング（前向き貪欲法）────────────────────
    seg_norms  = [norm_text(seg['text']) for seg in whisper_segments]
    line_norms = [norm_text(line) for line in fetched_lines]

    line_timing: dict = {}
    last_si = -1
    for li, ln in enumerate(line_norms):
        if not ln:
            continue
        best_score, best_si = 0.25, -1
        for si in range(last_si + 1, len(whisper_segments)):
            sn = seg_norms[si]
            if not sn:
                continue
            score = difflib.SequenceMatcher(None, sn, ln).ratio()
            if score > best_score:
                best_score, best_si = score, si
        if best_si >= 0:
            last_si = best_si
            seg = whisper_segments[best_si]
            line_timing[li] = (round(seg['start'], 3), round(seg['end'], 3))

    matched_lis = sorted(line_timing)
    matched_set = set(matched_lis)

    logger.info("アライメント: ラインマッチ %d/%d (ボーカル区間=%d個 計%.1fs)",
                len(line_timing), n_lines, len(voiced), total_vocal)

    # ── ゾーン補間 ────────────────────────────────────────────────────────
    all_timing: dict = {}

    def fill_zone(li_s: int, li_e: int, vt_s: float, vt_e: float):
        """[li_s, li_e) の行を ideal_vt ペースで [vt_s, vt_e) に配置。
        ゾーン内で時間が尽きたら残りの行はスキップ（テンポ維持優先）。"""
        if li_e <= li_s or vt_e <= vt_s:
            return
        for i in range(li_e - li_s):
            vt = vt_s + i * ideal_vt
            if vt >= vt_e:      # 時間切れ→残りスキップ
                break
            all_timing[li_s + i] = (
                round(_vt_to_abs(vt, voiced), 3),
                round(_vt_to_abs(min(vt + ideal_vt, vt_e), voiced), 3),
            )

    if not matched_lis:
        # マッチなし → 全体を均等配分
        fill_zone(0, n_lines, 0.0, total_vocal)
    else:
        first_li = matched_lis[0]
        last_li  = matched_lis[-1]

        # 先頭マッチ前
        if first_li > 0:
            fill_zone(0, first_li, 0.0, _abs_to_vt(line_timing[first_li][0], voiced))

        # マッチ行
        for li in matched_lis:
            all_timing[li] = line_timing[li]

        # マッチ間
        for a_li, b_li in zip(matched_lis, matched_lis[1:]):
            if b_li > a_li + 1:
                vt_a = _abs_to_vt(line_timing[a_li][1], voiced)
                vt_b = _abs_to_vt(line_timing[b_li][0], voiced)
                fill_zone(a_li + 1, b_li, vt_a, vt_b)

        # 末尾マッチ後（total_vocal を超えてよい：線形外挿で延長）
        if last_li < n_lines - 1:
            _, t_last = line_timing[last_li]
            vt_last = _abs_to_vt(t_last, voiced)
            for i in range(n_lines - last_li - 1):
                vt = vt_last + i * ideal_vt
                all_timing[last_li + 1 + i] = (
                    round(_vt_to_abs(vt, voiced), 3),
                    round(_vt_to_abs(vt + ideal_vt, voiced), 3),
                )

    # ── 結果を構築 ────────────────────────────────────────────────────────
    result      = []
    result_li   = []          # 各 result[i] に対応する li
    for li in range(n_lines):
        if li not in all_timing:
            continue
        t_s, t_e = all_timing[li]
        result.append({
            'start': t_s,
            'end':   t_e,
            'text':  fetched_lines[li],
            'words': make_words(fetched_lines[li], t_s, t_e),
        })
        result_li.append(li)

    # ── MIN_GAP：アンカー行は変更しない（カスケード遮断） ─────────────────
    MIN_GAP    = 0.8
    prev_start = -999.0
    for i, li in enumerate(result_li):
        if li in matched_set:
            # アンカー：時刻を保持し、前回基準をリセット
            prev_start = result[i]['start']
        else:
            if result[i]['start'] < prev_start + MIN_GAP:
                result[i]['start'] = round(prev_start + MIN_GAP, 3)
            prev_start = result[i]['start']

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
                lyrics = align_lyrics(fetched_lyrics, lyrics, pitch_data['duration'])
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
