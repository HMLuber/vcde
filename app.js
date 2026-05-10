/* Echtzeit-Objekterkennung — Visual Computing */

(() => {
  'use strict';

  /* ---------- DOM refs ---------------------------------- */
  const $ = (id) => document.getElementById(id);

  const video      = $('video');
  const overlay    = $('overlay');
  const ctx        = overlay.getContext('2d');
  const dropzone   = $('dropzone');
  const fileInput  = $('file-input');
  const loader     = $('loader');
  const loaderText = $('loader-text');
  const hud        = $('hud');
  const hudFps     = $('hud-fps');
  const controls   = $('controls');
  const playBtn    = $('play-btn');
  const playIcon   = $('play-icon');
  const pauseIcon  = $('pause-icon');
  const resetBtn   = $('reset-btn');
  const seek       = $('seek');
  const timeLabel  = $('time');
  const statusDot  = $('model-status-dot');
  const statusText = $('model-status-text');

  /* ---------- State ------------------------------------- */
  let model            = null;
  let modelPromise     = null; // singleton: prevent concurrent loads
  let rafId            = null;
  let running          = false;
  let currentObjectURL = null;

  const fpsWindow      = [];
  const FPS_WINDOW_SIZE = 30;
  let lastFrameTime    = 0;

  /* ---------- Colors ------------------------------------ */
  const PALETTE = [
    '#3b82f6','#eab308','#22c55e','#ec4899',
    '#06b6d4','#a855f7','#f97316','#10b981',
    '#ef4444','#8b5cf6',
  ];
  const CLASS_COLORS = {
    person: '#3b82f6', car: '#22c55e', truck: '#10b981',
    bus: '#06b6d4', bicycle: '#a855f7', motorcycle: '#ec4899',
    'traffic light': '#f97316', 'stop sign': '#ef4444',
  };

  function colorFor(cls) {
    if (CLASS_COLORS[cls]) return CLASS_COLORS[cls];
    let h = 0;
    for (let i = 0; i < cls.length; i++) h = (h * 31 + cls.charCodeAt(i)) | 0;
    return PALETTE[Math.abs(h) % PALETTE.length];
  }

  /* ---------- Model loading ----------------------------- */
  function bootModel() {
    // Return existing promise so concurrent callers all await the same load.
    if (modelPromise) return modelPromise;
    modelPromise = (async () => {
      try {
        await tf.setBackend('webgl');
        await tf.ready();
        model = await cocoSsd.load({ base: 'mobilenet_v2' });
        statusDot.classList.add('ready');
        statusText.textContent = 'Bereit';
      } catch (err) {
        console.error('Model load failed:', err);
        statusText.textContent = 'Fehler beim Laden';
        modelPromise = null; // allow retry on next attempt
      }
    })();
    return modelPromise;
  }

  /* ---------- Source handling --------------------------- */
  async function loadFile(file) {
    if (!file || !file.type.startsWith('video/')) {
      alert('Bitte eine Videodatei wählen (mp4, webm, mov, …).');
      return;
    }
    cleanupSource();
    currentObjectURL = URL.createObjectURL(file);
    video.srcObject = null;
    video.src = currentObjectURL;
    video.loop = true;
    video.muted = true;
    await prepareVideo();
  }

  function cleanupSource() {
    stopLoop();
    if (currentObjectURL) {
      URL.revokeObjectURL(currentObjectURL);
      currentObjectURL = null;
    }
    video.srcObject = null;
    video.removeAttribute('src');
    video.load();
  }

  async function prepareVideo() {
    await new Promise((resolve, reject) => {
      video.addEventListener('loadedmetadata', resolve, { once: true });
      video.addEventListener('error', () => reject(new Error('Video konnte nicht geladen werden.')), { once: true });
    }).catch((e) => { alert(e.message); throw e; });

    overlay.width  = video.videoWidth;
    overlay.height = video.videoHeight;
    fpsWindow.length = 0;

    dropzone.hidden  = true;
    hud.hidden       = false;
    controls.hidden  = false;

    try { await video.play(); } catch (_) {}

    if (!model) {
      loader.hidden    = false;
      loaderText.textContent = 'Modell wird geladen …';
      await bootModel();
      loader.hidden    = true;
    }

    startLoop();
    updateTime();
  }

  /* ---------- Detection loop ---------------------------- */
  function startLoop() {
    if (running) return;
    running = true;
    lastFrameTime = performance.now();
    rafId = requestAnimationFrame(tick);
  }

  function stopLoop() {
    running = false;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = null;
  }

  async function tick() {
    if (!running) return;

    if (video.readyState >= 2 && !video.paused && !video.ended && model) {
      let predictions = [];
      try { predictions = await model.detect(video); } catch (_) {}

      const now = performance.now();
      const dt  = now - lastFrameTime;
      lastFrameTime = now;
      if (dt > 0 && dt < 1000) {
        fpsWindow.push(1000 / dt);
        if (fpsWindow.length > FPS_WINDOW_SIZE) fpsWindow.shift();
      }

      const fps = fpsWindow.length
        ? fpsWindow.reduce((a, b) => a + b, 0) / fpsWindow.length
        : 0;
      hudFps.textContent = fps.toFixed(1);

      const frameArea = overlay.width * overlay.height;
      // Large vehicles (bus/truck) can legitimately fill ~40% of frame;
      // cars should never need more than ~12%. This suppresses the
      // "merged blob" false positive common in dense-traffic scenes.
      const maxAreaRatio = { bus: 0.40, truck: 0.40 };
      const filtered = predictions.filter((p) => {
        if (p.score < 0.3) return false;
        const [, , bw, bh] = p.bbox;
        const limit = maxAreaRatio[p.class] ?? 0.12;
        return (bw * bh) / frameArea <= limit;
      });
      drawOverlay(filtered);
    }

    rafId = requestAnimationFrame(tick);
  }

  /* ---------- Drawing ----------------------------------- */
  function drawOverlay(detections) {
    const w = overlay.width, h = overlay.height;
    ctx.clearRect(0, 0, w, h);

    const lineW    = Math.max(2, Math.round(Math.min(w, h) / 360));
    const fontSize = Math.max(12, Math.round(Math.min(w, h) / 40));
    ctx.font = `500 ${fontSize}px "JetBrains Mono", monospace`;
    ctx.textBaseline = 'top';

    for (const det of detections) {
      const [x, y, bw, bh] = det.bbox;
      const color = colorFor(det.class);
      const label = `${det.class} ${det.score.toFixed(2)}`;

      ctx.strokeStyle = color;
      ctx.lineWidth   = lineW;
      ctx.strokeRect(x, y, bw, bh);

      const padX = 4, padY = 3;
      const textW  = ctx.measureText(label).width;
      const labelH = fontSize + padY * 2;
      const labelY = Math.max(0, y - labelH);
      ctx.fillStyle = color;
      ctx.fillRect(x - lineW / 2, labelY, textW + padX * 2, labelH);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, x + padX - lineW / 2, labelY + padY);
    }
  }

  /* ---------- Controls / events ------------------------- */
  fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) loadFile(e.target.files[0]);
  });

  ['dragenter', 'dragover'].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add('dragging'); })
  );
  ['dragleave', 'drop'].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove('dragging'); })
  );
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
  });
  dropzone.addEventListener('click', (e) => {
    if (e.target.closest('label') || e.target.closest('input')) return;
    fileInput.click();
  });

  playBtn.addEventListener('click', () => { video.paused ? video.play() : video.pause(); });
  video.addEventListener('play',  () => { playIcon.hidden = true;  pauseIcon.hidden = false; });
  video.addEventListener('pause', () => { playIcon.hidden = false; pauseIcon.hidden = true;  });

  video.addEventListener('timeupdate', updateTime);
  function updateTime() {
    if (!isFinite(video.duration) || !video.duration) {
      timeLabel.textContent = '--:-- / --:--';
      seek.disabled = true;
      return;
    }
    seek.disabled = false;
    seek.value = (video.currentTime / video.duration) * 100;
    timeLabel.textContent = `${fmt(video.currentTime)} / ${fmt(video.duration)}`;
  }
  seek.addEventListener('input', () => {
    if (isFinite(video.duration)) video.currentTime = (seek.value / 100) * video.duration;
  });
  function fmt(s) {
    return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
  }

  resetBtn.addEventListener('click', () => {
    cleanupSource();
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    dropzone.hidden  = false;
    hud.hidden       = true;
    controls.hidden  = true;
  });

  /* ---------- Init -------------------------------------- */
  loader.hidden = true;
  bootModel();
})();
