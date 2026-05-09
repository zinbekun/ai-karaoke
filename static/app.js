'use strict';

// ── Constants ──────────────────────────────────────────────────────────────
const MIDI_MIN    = 45;   // A2  – lowest displayed pitch
const MIDI_MAX    = 84;   // C6  – highest displayed pitch
const MIDI_RANGE  = MIDI_MAX - MIDI_MIN;
const TIME_WINDOW = 9;    // seconds visible in canvas
const NOW_RATIO   = 0.28; // current-time line at 28% from left

const NOTE_NAMES  = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];

// Score thresholds (semitones)
const THRESH_PERFECT = 0.6;
const THRESH_GOOD    = 1.8;

// Points per frame (~60fps, so divide by frame interval ~10 hits/sec realistic)
const PTS_PERFECT = 15;
const PTS_GOOD    = 7;

// ── State ──────────────────────────────────────────────────────────────────
let analysisData   = null;
let audioBlob      = null;

let isPlaying      = false;
let isMicActive    = false;
let currentMidi    = null;   // user's live pitch
let animId         = null;
let frameIdx       = 0;

let score          = 0;
let totalFrames    = 0;
let hitFrames      = 0;
let lastEval       = '—';

// floating eval badges
const badges = [];  // {text, x, y, alpha, color}

// mic
let audioCtx  = null;
let analyser  = null;
let micBuffer = null;
let micStream = null;

// ── DOM ────────────────────────────────────────────────────────────────────
const audio        = document.getElementById('audio-player');
const canvas       = document.getElementById('pitch-canvas');
const ctx2d        = canvas.getContext('2d');

const uploadSec    = document.getElementById('upload-section');
const analyzingSec = document.getElementById('analyzing-section');
const karaokeSec   = document.getElementById('karaoke-section');
const newSongWrap  = document.getElementById('new-song-wrap');
const noPitchWarn  = document.getElementById('no-pitch-warning');

const fileInput    = document.getElementById('file-input');
const dropZone     = document.getElementById('drop-zone');
const playBtn      = document.getElementById('play-btn');
const micBtn       = document.getElementById('mic-btn');
const resetBtn     = document.getElementById('reset-btn');
const volumeSlider = document.getElementById('volume');
const newSongBtn   = document.getElementById('new-song-btn');

const songTitleEl  = document.getElementById('song-title');
const timeDisplay  = document.getElementById('time-display');
const scoreNum     = document.getElementById('score-num');
const evalText     = document.getElementById('eval-text');
const accuracyNum  = document.getElementById('accuracy-num');
const micNoteEl    = document.getElementById('mic-note');
const targetNoteEl = document.getElementById('target-note');

// ── File handling ──────────────────────────────────────────────────────────
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) loadFile(f);
});
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => { if (e.target.files[0]) loadFile(e.target.files[0]); });

async function loadFile(file) {
  audioBlob = file;
  audio.src = URL.createObjectURL(file);

  uploadSec.classList.add('hidden');
  analyzingSec.classList.remove('hidden');

  const fd = new FormData();
  fd.append('file', file);

  try {
    const res = await fetch('/api/analyze', { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail);
    }
    analysisData = await res.json();
  } catch (e) {
    alert('解析エラー: ' + e.message);
    analyzingSec.classList.add('hidden');
    uploadSec.classList.remove('hidden');
    return;
  }

  analyzingSec.classList.add('hidden');
  karaokeSec.classList.remove('hidden');
  newSongWrap.classList.remove('hidden');

  // 曲名表示（Shazam認識 or ファイル名）
  if (analysisData.song_info) {
    const { artist, title, source } = analysisData.song_info;
    songTitleEl.textContent = '♪ ' + (artist ? artist + ' - ' : '') + title;
  } else {
    songTitleEl.textContent = '♪ ' + (analysisData.filename || file.name);
  }
  noPitchWarn.classList.toggle('hidden', analysisData.segments.length > 0);

  resetState();
  resizeCanvas();
  startRenderLoop();
}

// ── Controls ───────────────────────────────────────────────────────────────
playBtn.addEventListener('click', () => {
  if (isPlaying) { audio.pause(); } else { audio.play(); }
});
audio.addEventListener('play',  () => { isPlaying = true;  playBtn.textContent = '⏸ 一時停止'; });
audio.addEventListener('pause', () => { isPlaying = false; playBtn.textContent = '▶ 再生'; });
audio.addEventListener('ended', () => { isPlaying = false; playBtn.textContent = '▶ 再生'; });

resetBtn.addEventListener('click', () => {
  audio.pause();
  audio.currentTime = 0;
  resetState();
});

volumeSlider.addEventListener('input', e => { audio.volume = +e.target.value; });

micBtn.addEventListener('click', () => { isMicActive ? stopMic() : startMic(); });

newSongBtn.addEventListener('click', () => {
  audio.pause();
  audio.src = '';
  stopMic();
  cancelAnimationFrame(animId);
  analysisData = null;
  karaokeSec.classList.add('hidden');
  newSongWrap.classList.add('hidden');
  uploadSec.classList.remove('hidden');
  fileInput.value = '';
});

// Clicking canvas seeks audio
canvas.addEventListener('click', e => {
  if (!analysisData) return;
  const rect = canvas.getBoundingClientRect();
  const ratio = (e.clientX - rect.left) / rect.width;
  const W = canvas.width;
  const nowX = W * NOW_RATIO;
  const pps  = W / TIME_WINDOW;
  const dt   = (ratio * W - nowX) / pps;
  audio.currentTime = Math.max(0, Math.min(analysisData.duration, audio.currentTime + dt));
});

function resetState() {
  isPlaying  = false;
  score      = 0;
  totalFrames = 0;
  hitFrames  = 0;
  lastEval   = '—';
  badges.length = 0;
  playBtn.textContent = '▶ 再生';
  updateHUD(null, null);
}

// ── Microphone pitch detection ─────────────────────────────────────────────
async function startMic() {
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    audioCtx  = new AudioContext();
    const src = audioCtx.createMediaStreamSource(micStream);
    analyser  = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    src.connect(analyser);
    micBuffer = new Float32Array(analyser.fftSize);
    isMicActive = true;
    micBtn.textContent = 'マイク OFF';
    micBtn.classList.add('mic-active');
  } catch (e) {
    alert('マイクへのアクセスができません: ' + e.message);
  }
}

function stopMic() {
  micStream?.getTracks().forEach(t => t.stop());
  audioCtx?.close();
  micStream = audioCtx = analyser = micBuffer = null;
  isMicActive = false;
  currentMidi = null;
  micBtn.textContent = 'マイク ON';
  micBtn.classList.remove('mic-active');
}

// McLeod Pitch Method (NSDF) – runs every 6 render frames (~10 Hz)
function detectPitch() {
  if (!analyser || !micBuffer) return null;
  analyser.getFloatTimeDomainData(micBuffer);

  const buf = micBuffer;
  const N   = buf.length;
  const HALF = N >> 1;

  // RMS gate
  let rms = 0;
  for (let i = 0; i < N; i++) rms += buf[i] * buf[i];
  if (Math.sqrt(rms / N) < 0.008) return null;

  const sr       = audioCtx.sampleRate;
  const minOff   = Math.ceil(sr / 1400);  // ~1400 Hz max
  const maxOff   = Math.floor(sr / 75);   // ~75 Hz min

  let bestOff = -1, bestVal = 0;

  for (let tau = minOff; tau < Math.min(maxOff, HALF); tau++) {
    let numer = 0, denom = 0;
    for (let i = 0; i < HALF; i++) {
      const a = buf[i], b = buf[i + tau];
      numer += a * b;
      denom += a * a + b * b;
    }
    if (denom === 0) continue;
    const r = 2 * numer / denom;
    if (r > bestVal) { bestVal = r; bestOff = tau; }
  }

  if (bestVal < 0.65 || bestOff === -1) return null;

  // Parabolic interpolation
  const y1 = bestOff > 1 ? nsdf(buf, HALF, bestOff - 1) : bestVal;
  const y2 = bestVal;
  const y3 = bestOff < HALF - 1 ? nsdf(buf, HALF, bestOff + 1) : bestVal;
  const d  = 2 * (2 * y2 - y1 - y3);
  const refinedOff = d !== 0 ? bestOff + (y1 - y3) / d : bestOff;

  const hz = sr / refinedOff;
  if (hz < 60 || hz > 1600) return null;
  return 12 * Math.log2(hz / 440) + 69;
}

function nsdf(buf, half, tau) {
  let numer = 0, denom = 0;
  for (let i = 0; i < half; i++) {
    const a = buf[i], b = buf[i + tau];
    numer += a * b;
    denom += a * a + b * b;
  }
  return denom === 0 ? 0 : 2 * numer / denom;
}

// ── Helpers ────────────────────────────────────────────────────────────────
function midiToY(midi) {
  return canvas.height * (1 - (midi - MIDI_MIN) / MIDI_RANGE);
}

function midiToNoteName(midi) {
  const m = Math.round(midi);
  const oct = Math.floor(m / 12) - 1;
  return NOTE_NAMES[m % 12] + oct;
}

function getTargetAt(t) {
  if (!analysisData) return null;
  for (const s of analysisData.segments) {
    if (t >= s.start && t <= s.end) return s;
  }
  return null;
}

function formatTime(t) {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

// ── Canvas resize ──────────────────────────────────────────────────────────
function resizeCanvas() {
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width  = Math.floor(rect.width);
  canvas.height = 400;
}
window.addEventListener('resize', resizeCanvas);

// ── Render loop ────────────────────────────────────────────────────────────
function startRenderLoop() {
  cancelAnimationFrame(animId);
  frameIdx = 0;
  loop();
}

function loop() {
  animId = requestAnimationFrame(loop);
  frameIdx++;

  if (isMicActive && frameIdx % 6 === 0) {
    currentMidi = detectPitch();
  }

  const t      = audio.currentTime;
  const target = getTargetAt(t);

  if (isPlaying && isMicActive && target) {
    scoreFrame(target);
  }

  drawCanvas(t, target);
  updateHUD(target, currentMidi);
}

// ── Scoring ────────────────────────────────────────────────────────────────
function scoreFrame(target) {
  totalFrames++;
  if (currentMidi === null) { setEval('MISS', '#f87171'); return; }
  const diff = Math.abs(currentMidi - target.midi);
  if (diff <= THRESH_PERFECT) {
    hitFrames++;
    score += PTS_PERFECT;
    setEval('PERFECT!', '#facc15');
    if (frameIdx % 30 === 0) spawnBadge('PERFECT!', '#facc15');
  } else if (diff <= THRESH_GOOD) {
    hitFrames++;
    score += PTS_GOOD;
    setEval('GOOD', '#4ade80');
    if (frameIdx % 40 === 0) spawnBadge('GOOD', '#4ade80');
  } else {
    setEval('MISS', '#f87171');
  }
}

let lastEvalStr = '';
function setEval(s, c) {
  lastEval = s;
  if (lastEvalStr !== s) {
    evalText.textContent = s;
    evalText.style.color = c;
    lastEvalStr = s;
  }
}

function spawnBadge(text, color) {
  const nowX = canvas.width * NOW_RATIO;
  badges.push({ text, color, x: nowX + 10, y: canvas.height * 0.4, alpha: 1 });
}

// ── HUD update ─────────────────────────────────────────────────────────────
function updateHUD(target, userMidi) {
  scoreNum.textContent = score.toLocaleString();
  if (totalFrames > 0) {
    accuracyNum.textContent = Math.round(hitFrames / totalFrames * 100) + '%';
  }
  micNoteEl.textContent    = userMidi !== null ? midiToNoteName(userMidi) : '—';
  targetNoteEl.textContent = target ? target.note : '—';
  timeDisplay.textContent  = analysisData
    ? `${formatTime(audio.currentTime)} / ${formatTime(analysisData.duration)}`
    : '0:00 / 0:00';
}

// ── Canvas draw ────────────────────────────────────────────────────────────
function drawCanvas(currentTime, target) {
  const W   = canvas.width;
  const H   = canvas.height;
  const pps = W / TIME_WINDOW;
  const nowX = W * NOW_RATIO;

  // Background
  ctx2d.fillStyle = '#03030f';
  ctx2d.fillRect(0, 0, W, H);

  // Grid: horizontal lines at each C note
  ctx2d.lineWidth = 1;
  for (let midi = MIDI_MIN; midi <= MIDI_MAX; midi++) {
    const y = midiToY(midi);
    const nc = midi % 12;
    if (nc === 0) {
      // C note: solid line + label
      ctx2d.strokeStyle = '#1a1a40';
      ctx2d.setLineDash([]);
      ctx2d.beginPath(); ctx2d.moveTo(0, y); ctx2d.lineTo(W, y); ctx2d.stroke();
      ctx2d.fillStyle = '#35355a';
      ctx2d.font = '10px monospace';
      ctx2d.fillText(`C${Math.floor(midi / 12) - 1}`, 4, y - 3);
    } else if (nc === 7) {
      // G note: dashed minor line
      ctx2d.strokeStyle = '#0f0f28';
      ctx2d.setLineDash([3, 8]);
      ctx2d.beginPath(); ctx2d.moveTo(0, y); ctx2d.lineTo(W, y); ctx2d.stroke();
    }
  }
  ctx2d.setLineDash([]);

  if (!analysisData) return;

  const semH = H / MIDI_RANGE;  // pixel height per semitone

  // ── Target pitch bars ──
  for (const seg of analysisData.segments) {
    const x1 = nowX + (seg.start - currentTime) * pps;
    const x2 = nowX + (seg.end   - currentTime) * pps;
    if (x2 < -4 || x1 > W + 4) continue;

    const y   = midiToY(seg.midi);
    const bH  = semH * 0.85;
    const bX  = Math.max(0, x1) + 1;
    const bW  = Math.min(W, x2) - bX - 1;
    if (bW <= 0) continue;

    const isPast    = seg.end   < currentTime;
    const isActive  = seg.start <= currentTime && currentTime <= seg.end;

    let fillColor;
    if (isActive) {
      fillColor = createGrad(ctx2d, bX, bX + bW, '#c084fc', '#818cf8');
    } else if (isPast) {
      fillColor = '#1e1e44';
    } else {
      fillColor = createGrad(ctx2d, bX, bX + bW, '#1e3a6a', '#1e4a8a');
    }

    ctx2d.fillStyle = fillColor;
    if (isActive) {
      ctx2d.shadowColor = '#a78bfa';
      ctx2d.shadowBlur  = 18;
    }
    roundRect2d(ctx2d, bX, y - bH / 2, bW, bH, 4);
    ctx2d.fill();
    ctx2d.shadowBlur = 0;

    // Note label on bar
    if (bW > 28) {
      ctx2d.fillStyle = 'rgba(255,255,255,.65)';
      ctx2d.font = 'bold 9px monospace';
      ctx2d.fillText(seg.note, bX + 4, y + 3);
    }
  }

  // ── Raw pitch curve (faint) ──
  if (analysisData.raw_pitch.length) {
    ctx2d.save();
    ctx2d.strokeStyle = 'rgba(96,165,250,.2)';
    ctx2d.lineWidth = 1.5;
    ctx2d.setLineDash([2, 5]);
    ctx2d.beginPath();
    let started = false;
    for (const p of analysisData.raw_pitch) {
      const x = nowX + (p.t - currentTime) * pps;
      if (x < 0 || x > W) { started = false; continue; }
      const y = midiToY(p.midi);
      if (!started) { ctx2d.moveTo(x, y); started = true; } else { ctx2d.lineTo(x, y); }
    }
    ctx2d.stroke();
    ctx2d.setLineDash([]);
    ctx2d.restore();
  }

  // ── "Now" vertical line ──
  ctx2d.strokeStyle = 'rgba(255,255,255,.5)';
  ctx2d.lineWidth = 2;
  ctx2d.beginPath();
  ctx2d.moveTo(nowX, 0);
  ctx2d.lineTo(nowX, H);
  ctx2d.stroke();
  // Triangle at top
  ctx2d.fillStyle = 'rgba(255,255,255,.8)';
  ctx2d.beginPath();
  ctx2d.moveTo(nowX, 10);
  ctx2d.lineTo(nowX - 6, 0);
  ctx2d.lineTo(nowX + 6, 0);
  ctx2d.closePath();
  ctx2d.fill();

  // ── User pitch indicator ──
  if (isMicActive && currentMidi !== null) {
    const clampedMidi = Math.max(MIDI_MIN, Math.min(MIDI_MAX, currentMidi));
    const y = midiToY(clampedMidi);
    const isHit = target && Math.abs(currentMidi - target.midi) <= THRESH_GOOD;
    const col   = isHit ? '#4ade80' : '#facc15';

    ctx2d.shadowColor = col;
    ctx2d.shadowBlur  = 24;

    // Horizontal trail
    ctx2d.fillStyle = col + '44';
    ctx2d.fillRect(0, y - 3, nowX, 6);

    // Circle
    ctx2d.fillStyle = col;
    ctx2d.beginPath();
    ctx2d.arc(nowX, y, 9, 0, Math.PI * 2);
    ctx2d.fill();

    ctx2d.shadowBlur = 0;

    // Note label
    ctx2d.fillStyle = col;
    ctx2d.font = 'bold 12px monospace';
    ctx2d.fillText(midiToNoteName(currentMidi), nowX + 14, y + 5);

    // Perfect flash
    if (target && Math.abs(currentMidi - target.midi) <= THRESH_PERFECT && isPlaying) {
      ctx2d.fillStyle = 'rgba(74,222,128,.06)';
      ctx2d.fillRect(0, 0, nowX, H);
    }
  }

  // ── Floating badges ──
  for (let i = badges.length - 1; i >= 0; i--) {
    const b = badges[i];
    ctx2d.globalAlpha = b.alpha;
    ctx2d.fillStyle = b.color;
    ctx2d.font = 'bold 18px sans-serif';
    ctx2d.fillText(b.text, b.x, b.y);
    ctx2d.globalAlpha = 1;
    b.y -= 0.8;
    b.alpha -= 0.018;
    if (b.alpha <= 0) badges.splice(i, 1);
  }

  // ── Time axis ticks ──
  ctx2d.fillStyle = '#35355a';
  ctx2d.font = '10px monospace';
  for (let dt = -Math.ceil(TIME_WINDOW * NOW_RATIO); dt < TIME_WINDOW; dt += 2) {
    const t = Math.round(currentTime + dt);
    if (t < 0 || t > analysisData.duration) continue;
    const x = nowX + dt * pps;
    if (x < 0 || x > W) continue;
    ctx2d.fillText(formatTime(t), x + 2, H - 4);
    ctx2d.strokeStyle = '#1a1a40';
    ctx2d.lineWidth = 1;
    ctx2d.beginPath(); ctx2d.moveTo(x, H - 14); ctx2d.lineTo(x, H - 1); ctx2d.stroke();
  }
}

// ── Canvas helpers ─────────────────────────────────────────────────────────
function createGrad(c, x1, x2, col1, col2) {
  const g = c.createLinearGradient(x1, 0, x2, 0);
  g.addColorStop(0, col1);
  g.addColorStop(1, col2);
  return g;
}

function roundRect2d(c, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  c.beginPath();
  c.moveTo(x + r, y);
  c.lineTo(x + w - r, y);
  c.quadraticCurveTo(x + w, y,     x + w, y + r);
  c.lineTo(x + w, y + h - r);
  c.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  c.lineTo(x + r,     y + h);
  c.quadraticCurveTo(x,     y + h, x,     y + h - r);
  c.lineTo(x,     y + r);
  c.quadraticCurveTo(x,     y,     x + r, y);
  c.closePath();
}
