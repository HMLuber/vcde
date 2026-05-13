/* Echtzeit-Objekterkennung — Visual Computing
   Main thread: UI + pre-processing only.
   Inference runs in worker.js to keep the tab responsive. */

(() => {
  'use strict';

  let INPUT_SIZE = 160; // updated from worker 'ready' to match the loaded model

  /* ---- DOM refs ---------------------------------------- */
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

  /* ---- State ------------------------------------------- */
  let rafId            = null;
  let running          = false;
  let currentObjectURL = null;
  let workerReady      = false;
  let inferring        = false;

  // Reused across frames — avoids repeated allocations
  const offscreen  = document.createElement('canvas');
  offscreen.width  = INPUT_SIZE;
  offscreen.height = INPUT_SIZE;
  const offCtx     = offscreen.getContext('2d', { willReadFrequently: true });

  const fpsWindow       = [];
  const FPS_WINDOW_SIZE = 20;
  let lastResultTime    = 0;

  /* ---- Colors ------------------------------------------ */
  const PALETTE = [
    '#3b82f6','#eab308','#22c55e','#ec4899','#06b6d4',
    '#a855f7','#f97316','#10b981','#ef4444','#8b5cf6',
  ];
  const CLASS_COLORS = {
    person: '#3b82f6', car: '#22c55e', truck: '#10b981',
    bus: '#06b6d4', bicycle: '#a855f7', motorcycle: '#ec4899',
    'traffic light': '#f97316', 'stop sign': '#ef4444',
    'sports ball': '#eab308',
  };
  function colorFor(cls) {
    if (CLASS_COLORS[cls]) return CLASS_COLORS[cls];
    let h = 0;
    for (const c of cls) h = (h * 31 + c.charCodeAt(0)) | 0;
    return PALETTE[Math.abs(h) % PALETTE.length];
  }

  /* ---- Web Worker -------------------------------------- */
  const worker = new Worker('worker.js');

  worker.onmessage = ({ data }) => {
    if (data.type === 'ready') {
      // Sync INPUT_SIZE and offscreen canvas to whatever size the model actually uses
      INPUT_SIZE = data.inputSize;
      offscreen.width = offscreen.height = INPUT_SIZE;
      workerReady = true;
      statusDot.classList.add('ready');
      const providerLabel = data.provider === 'WASM' && data.numThreads > 1
        ? `WASM · ${data.numThreads} Threads`
        : data.provider;
      statusText.textContent = `Bereit · YOLOv8n · ${providerLabel}`;
      loader.hidden = true;
      if (data.gpuError) console.warn('WebGPU nicht verfügbar:', data.gpuError);
    }

    if (data.type === 'result') {
      inferring = false;
      if (!running) return;
      const now = performance.now();
      const dt  = now - lastResultTime;
      if (lastResultTime > 0 && dt > 0 && dt < 5000) {
        fpsWindow.push(1000 / dt);
        if (fpsWindow.length > FPS_WINDOW_SIZE) fpsWindow.shift();
      }
      lastResultTime = now;
      const fps = fpsWindow.length
        ? fpsWindow.reduce((a, b) => a + b) / fpsWindow.length : 0;
      hudFps.textContent = fps.toFixed(1);
      drawOverlay(data.detections);
    }

    if (data.type === 'error') {
      statusText.textContent = 'Fehler beim Laden';
      console.error('Worker:', data.message);
    }
  };

  worker.onerror = (e) => {
    statusText.textContent = 'Worker-Fehler';
    console.error('Worker error:', e);
  };

  /* ---- Pre-processing: letterbox → CHW float32 --------- */
  function preprocess(videoEl) {
    const vw    = videoEl.videoWidth,  vh = videoEl.videoHeight;
    const scale = Math.min(INPUT_SIZE / vw, INPUT_SIZE / vh);
    const newW  = Math.round(vw * scale), newH = Math.round(vh * scale);
    const padX  = Math.floor((INPUT_SIZE - newW) / 2);
    const padY  = Math.floor((INPUT_SIZE - newH) / 2);

    offCtx.fillStyle = '#808080';
    offCtx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
    offCtx.drawImage(videoEl, padX, padY, newW, newH);

    const { data } = offCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const float32  = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    const area     = INPUT_SIZE * INPUT_SIZE;
    for (let i = 0; i < area; i++) {
      float32[i]          = data[i * 4]     / 255; // R
      float32[i + area]   = data[i * 4 + 1] / 255; // G
      float32[i + 2*area] = data[i * 4 + 2] / 255; // B
    }
    return { float32, scale, padX, padY };
  }

  /* ---- Source handling --------------------------------- */
  async function loadFile(file) {
    if (!file || !file.type.startsWith('video/')) {
      alert('Bitte eine Videodatei wählen (mp4, webm, mov, …).');
      return;
    }
    cleanupSource();
    currentObjectURL = URL.createObjectURL(file);
    video.srcObject  = null;
    video.src        = currentObjectURL;
    video.loop       = true;
    video.muted      = true;
    await prepareVideo();
  }

  function cleanupSource() {
    stopLoop();
    inferring = false;
    if (currentObjectURL) { URL.revokeObjectURL(currentObjectURL); currentObjectURL = null; }
    video.srcObject = null;
    video.removeAttribute('src');
    video.load();
    ctx.clearRect(0, 0, overlay.width, overlay.height);
  }

  async function prepareVideo() {
    await new Promise((resolve, reject) => {
      video.addEventListener('loadedmetadata', resolve, { once: true });
      video.addEventListener('error', () =>
        reject(new Error('Video konnte nicht geladen werden.')), { once: true });
    }).catch((e) => { alert(e.message); throw e; });

    overlay.width    = video.videoWidth;
    overlay.height   = video.videoHeight;
    fpsWindow.length = 0;
    lastResultTime   = 0;

    dropzone.hidden = true;
    hud.hidden      = false;
    controls.hidden = false;

    try { await video.play(); } catch (_) {}

    if (!workerReady) {
      loader.hidden          = false;
      loaderText.textContent = 'YOLOv8n wird geladen … (320 px · ~12 MB)';
    }

    startLoop();
    updateTime();
  }

  /* ---- Detection loop (RAF — no awaiting) -------------- */
  function startLoop() {
    if (running) return;
    running = true;
    rafId   = requestAnimationFrame(tick);
  }

  function stopLoop() {
    running = false;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = null;
  }

  function tick() {
    if (!running) return;
    if (video.readyState >= 2 && !video.paused && !video.ended && workerReady && !inferring) {
      inferring = true;
      try {
        const { float32, scale, padX, padY } = preprocess(video);
        worker.postMessage(
          { type: 'infer', float32, scale, padX, padY,
            origW: video.videoWidth, origH: video.videoHeight },
          [float32.buffer],
        );
      } catch (err) {
        inferring = false;
        console.error('preprocess failed:', err);
      }
    }
    rafId = requestAnimationFrame(tick);
  }

  /* ---- Drawing ----------------------------------------- */
  function drawOverlay(detections) {
    const w = overlay.width, h = overlay.height;
    ctx.clearRect(0, 0, w, h);

    const lineW    = Math.max(2, Math.round(Math.min(w, h) / 360));
    const fontSize = Math.max(12, Math.round(Math.min(w, h) / 40));
    ctx.font         = `500 ${fontSize}px "JetBrains Mono", monospace`;
    ctx.textBaseline = 'top';

    for (const det of detections) {
      const [x, y, bw, bh] = det.bbox;
      const color = colorFor(det.class);
      const label = `${det.class} ${det.score.toFixed(2)}`;

      ctx.strokeStyle = color;
      ctx.lineWidth   = lineW;
      ctx.strokeRect(x, y, bw, bh);

      const px = 4, py = 3;
      const textW  = ctx.measureText(label).width;
      const labelH = fontSize + py * 2;
      const labelY = Math.max(0, y - labelH);
      ctx.fillStyle = color;
      ctx.fillRect(x - lineW / 2, labelY, textW + px * 2, labelH);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, x + px - lineW / 2, labelY + py);
    }
  }

  /* ---- Controls / events ------------------------------- */
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
    if (!e.target.closest('label') && !e.target.closest('input')) fileInput.click();
  });

  playBtn.addEventListener('click', () => { video.paused ? video.play() : video.pause(); });
  video.addEventListener('play',  () => { playIcon.hidden = true;  pauseIcon.hidden = false; });
  video.addEventListener('pause', () => { playIcon.hidden = false; pauseIcon.hidden = true;  });

  video.addEventListener('timeupdate', updateTime);
  function updateTime() {
    if (!isFinite(video.duration) || !video.duration) {
      timeLabel.textContent = '--:-- / --:--'; seek.disabled = true; return;
    }
    seek.disabled         = false;
    seek.value            = (video.currentTime / video.duration) * 100;
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
    dropzone.hidden = false;
    hud.hidden      = true;
    controls.hidden = true;
  });
})();
