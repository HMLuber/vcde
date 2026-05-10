/* =========================================================
   Echtzeit-Objekterkennung — Visual Computing
   Browser-based real-time object detection on uploaded video
   ========================================================= */

(() => {
  'use strict';

  /* ---------- DOM refs ---------------------------------- */
  const $ = (id) => document.getElementById(id);

  const video       = $('video');
  const overlay     = $('overlay');
  const ctx         = overlay.getContext('2d');
  const dropzone    = $('dropzone');
  const fileInput   = $('file-input');
  const webcamBtn   = $('webcam-btn');
  const sampleBtn   = $('sample-btn');
  const loader      = $('loader');
  const loaderText  = $('loader-text');
  const hud         = $('hud');
  const hudFps      = $('hud-fps');
  const hudInfer    = $('hud-infer');
  const hudObjects  = $('hud-objects');
  const controls    = $('controls');
  const playBtn     = $('play-btn');
  const playIcon    = $('play-icon');
  const pauseIcon   = $('pause-icon');
  const resetBtn    = $('reset-btn');
  const seek        = $('seek');
  const timeLabel   = $('time');
  const fpsValue    = $('fps-value');
  const fpsNeedle   = $('fps-needle');
  const inferValue  = $('infer-value');
  const resValue    = $('res-value');
  const backendVal  = $('backend-value');
  const framesValue = $('frames-value');
  const classList   = $('class-list');
  const classCounter= $('class-counter');
  const thresholdEl = $('threshold');
  const thresholdDisplay = $('threshold-display');
  const statusDot   = $('model-status-dot');
  const statusText  = $('model-status-text');

  /* ---------- State ------------------------------------- */
  let model = null;
  let rafId = null;
  let running = false;
  let activeStream = null;          // for webcam tracks
  let currentObjectURL = null;      // for uploaded video blobs

  let threshold = parseFloat(thresholdEl.value);

  // Rolling FPS calculation
  const fpsWindow = [];
  const FPS_WINDOW_SIZE = 30;
  let lastFrameTime = 0;
  let totalFrames = 0;

  // Class-level stats: { className: { count, lastConf, color } }
  const classStats = new Map();

  /* ---------- Color palette for bounding boxes ---------- */
  // Match the screenshot vibe: blue for person, yellow for tie, etc.
  // Stable per-class colors via a hash-based picker.
  const PALETTE = [
    '#3b82f6', // blue
    '#eab308', // yellow
    '#22c55e', // green
    '#ec4899', // pink
    '#06b6d4', // cyan
    '#a855f7', // purple
    '#f97316', // orange
    '#10b981', // emerald
    '#ef4444', // red
    '#8b5cf6', // violet
    '#14b8a6', // teal
    '#f59e0b', // amber
  ];
  const PRIORITY_COLORS = {
    person: '#3b82f6',
    tie: '#eab308',
    car: '#22c55e',
    truck: '#10b981',
    bus: '#06b6d4',
    bicycle: '#a855f7',
    motorcycle: '#ec4899',
    'traffic light': '#f97316',
    'stop sign': '#ef4444',
  };
  function hashStr(s) {
    let h = 0;
    for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
    return Math.abs(h);
  }
  function colorFor(cls) {
    if (PRIORITY_COLORS[cls]) return PRIORITY_COLORS[cls];
    return PALETTE[hashStr(cls) % PALETTE.length];
  }

  /* ---------- Boot: load model -------------------------- */
  async function bootModel() {
    try {
      // Set TF backend explicitly to WebGL for speed
      await tf.setBackend('webgl');
      await tf.ready();
      backendVal.textContent = tf.getBackend().toUpperCase();

      loaderText.textContent = 'Modell wird geladen … (~ 27 MB)';
      // 'lite_mobilenet_v2' is fastest; keep default for accuracy/speed balance.
      model = await cocoSsd.load({ base: 'lite_mobilenet_v2' });

      statusDot.classList.add('ready');
      statusText.textContent = 'Bereit';
    } catch (err) {
      console.error('Model load failed:', err);
      statusText.textContent = 'Fehler beim Laden';
      loaderText.textContent = 'Modell konnte nicht geladen werden. Internet­verbindung?';
    }
  }

  /* ---------- Source handling --------------------------- */
  async function loadFile(file) {
    if (!file || !file.type.startsWith('video/')) {
      alert('Bitte eine Videodatei auswählen (mp4, webm, mov, …).');
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

  async function loadWebcam() {
    cleanupSource();
    try {
      activeStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' },
        audio: false,
      });
      video.src = '';
      video.srcObject = activeStream;
      video.loop = false;
      video.muted = true;
      await prepareVideo();
    } catch (err) {
      console.error(err);
      alert('Kamera-Zugriff wurde abgelehnt oder ist nicht verfügbar.');
    }
  }

  async function loadSample() {
    // A small public-domain traffic clip from W3C's sample collection.
    // If unreachable (offline), the user gets a clear message.
    const SAMPLES = [
      // Coverr provides free CC0 clips; this is a small one.
      'https://storage.googleapis.com/tfjs-models/assets/posenet/coco_dance.mp4',
    ];
    cleanupSource();
    video.srcObject = null;
    video.crossOrigin = 'anonymous';
    video.src = SAMPLES[0];
    video.loop = true;
    video.muted = true;
    await prepareVideo();
  }

  function cleanupSource() {
    stopLoop();
    if (activeStream) {
      activeStream.getTracks().forEach((t) => t.stop());
      activeStream = null;
    }
    if (currentObjectURL) {
      URL.revokeObjectURL(currentObjectURL);
      currentObjectURL = null;
    }
    video.srcObject = null;
    video.removeAttribute('src');
    video.load();
  }

  async function prepareVideo() {
    // Wait for metadata so we know the natural size
    await new Promise((resolve, reject) => {
      const onMeta = () => { video.removeEventListener('loadedmetadata', onMeta); resolve(); };
      const onErr  = () => { video.removeEventListener('error', onErr); reject(new Error('Video-Quelle konnte nicht geladen werden.')); };
      video.addEventListener('loadedmetadata', onMeta, { once: true });
      video.addEventListener('error', onErr, { once: true });
    }).catch((e) => {
      alert(e.message + '\nFalls es das Demo-Video ist: prüfe deine Verbindung.');
      throw e;
    });

    // Match the canvas to the video's intrinsic resolution
    overlay.width  = video.videoWidth;
    overlay.height = video.videoHeight;
    resValue.textContent = `${video.videoWidth} × ${video.videoHeight}`;

    // Reset stats
    fpsWindow.length = 0;
    classStats.clear();
    totalFrames = 0;
    renderClassList();

    // UI states
    dropzone.style.display = 'none';
    hud.hidden = false;
    controls.hidden = false;
    if (loader) loader.hidden = true;

    try { await video.play(); } catch (_) { /* autoplay block on some browsers */ }

    if (!model) {
      loader.hidden = false;
      loaderText.textContent = 'Modell wird geladen …';
      await bootModel();
      loader.hidden = true;
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
      const t0 = performance.now();
      let predictions = [];
      try {
        predictions = await model.detect(video);
      } catch (e) {
        // continue; one frame failure shouldn't kill the loop
      }
      const t1 = performance.now();

      // Filter by confidence threshold
      const filtered = predictions.filter((p) => p.score >= threshold);

      // Update FPS window using full frame time (RAF cadence)
      const now = performance.now();
      const dt = now - lastFrameTime;
      lastFrameTime = now;
      if (dt > 0 && dt < 1000) {
        fpsWindow.push(1000 / dt);
        if (fpsWindow.length > FPS_WINDOW_SIZE) fpsWindow.shift();
      }

      const inferMs = t1 - t0;
      totalFrames++;

      drawOverlay(filtered);
      updateClassStats(filtered);
      updateTelemetry(inferMs, filtered.length);
    }

    rafId = requestAnimationFrame(tick);
  }

  /* ---------- Drawing ----------------------------------- */
  function drawOverlay(detections) {
    const w = overlay.width;
    const h = overlay.height;
    ctx.clearRect(0, 0, w, h);

    // Drawing scale: line width relative to video size
    const lineW = Math.max(2, Math.round(Math.min(w, h) / 360));
    const fontSize = Math.max(12, Math.round(Math.min(w, h) / 40));
    ctx.font = `500 ${fontSize}px "JetBrains Mono", monospace`;
    ctx.textBaseline = 'top';

    for (const det of detections) {
      const [x, y, bw, bh] = det.bbox;
      const color = colorFor(det.class);
      const label = `${det.class} ${det.score.toFixed(2)}`;

      // Box
      ctx.strokeStyle = color;
      ctx.lineWidth = lineW;
      ctx.strokeRect(x, y, bw, bh);

      // Label background
      const padX = 4, padY = 3;
      const textW = ctx.measureText(label).width;
      const labelH = fontSize + padY * 2;
      const labelY = Math.max(0, y - labelH);
      ctx.fillStyle = color;
      ctx.fillRect(x - lineW / 2, labelY, textW + padX * 2, labelH);

      // Label text
      ctx.fillStyle = '#0b0c0a';
      ctx.fillText(label, x + padX - lineW / 2, labelY + padY);
    }
  }

  /* ---------- Class stats ------------------------------- */
  function updateClassStats(detections) {
    // Snapshot the current frame: tally counts and average confidence per class
    const frameCounts = new Map();
    for (const d of detections) {
      const e = frameCounts.get(d.class) || { count: 0, conf: 0 };
      e.count++;
      e.conf = Math.max(e.conf, d.score);
      frameCounts.set(d.class, e);
    }
    // Merge into running stats (keep current-frame count, EMA for confidence)
    for (const [cls, e] of frameCounts.entries()) {
      const prev = classStats.get(cls) || { count: 0, lastConf: e.conf, peak: 0 };
      prev.count = e.count;
      prev.lastConf = prev.lastConf * 0.7 + e.conf * 0.3; // light smoothing
      prev.peak = Math.max(prev.peak, e.conf);
      classStats.set(cls, prev);
    }
    // Decay classes not seen this frame
    for (const [cls, prev] of classStats.entries()) {
      if (!frameCounts.has(cls)) {
        prev.count = 0;
      }
    }

    if (totalFrames % 6 === 0) renderClassList(); // throttle DOM updates to ~10Hz
  }

  function renderClassList() {
    // Sort: classes currently present first (desc by count), then others
    const entries = [...classStats.entries()]
      .sort((a, b) => (b[1].count - a[1].count) || (b[1].peak - a[1].peak));

    if (entries.length === 0) {
      classList.innerHTML = '<li class="class-empty">Noch keine Objekte erkannt.</li>';
      classCounter.textContent = '0';
      return;
    }
    classCounter.textContent = entries.filter(([, v]) => v.count > 0).length;

    const html = entries.slice(0, 30).map(([cls, v]) => {
      const c = colorFor(cls);
      const isLive = v.count > 0;
      const opacity = isLive ? 1 : 0.4;
      return `
        <li class="class-item" style="border-left-color:${c}; opacity:${opacity}">
          <span class="class-swatch" style="background:${c}"></span>
          <span class="class-name">${escapeHtml(cls)}</span>
          <span class="class-count">×${v.count}</span>
          <span class="class-conf">${(v.lastConf * 100).toFixed(0)}%</span>
        </li>
      `;
    }).join('');
    classList.innerHTML = html;
  }

  function escapeHtml(s) { return s.replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

  /* ---------- Telemetry --------------------------------- */
  function updateTelemetry(inferMs, objCount) {
    const fps = fpsWindow.length
      ? fpsWindow.reduce((a, b) => a + b, 0) / fpsWindow.length
      : 0;

    // HUD
    hudFps.textContent     = fps.toFixed(1);
    hudInfer.textContent   = `${inferMs.toFixed(0)} ms`;
    hudObjects.textContent = objCount;

    // Big metric
    const intPart = Math.floor(fps);
    const decPart = ((fps - intPart) * 10) | 0;
    fpsValue.innerHTML = `${intPart}<small>.${decPart}</small>`;

    fpsValue.classList.remove('under', 'target', 'over');
    if (fps >= 30)      fpsValue.classList.add('over');
    else if (fps >= 24) fpsValue.classList.add('target');
    else                fpsValue.classList.add('under');

    // Needle: 0 → 0%, 60+ → 100%
    const pct = Math.max(0, Math.min(100, (fps / 60) * 100));
    fpsNeedle.style.left = `${pct}%`;

    inferValue.textContent  = `${inferMs.toFixed(1)} ms`;
    framesValue.textContent = totalFrames.toLocaleString('de-DE');
  }

  /* ---------- Controls / events ------------------------- */
  // File picker
  fileInput.addEventListener('change', (e) => {
    const f = e.target.files[0];
    if (f) loadFile(f);
  });

  // Webcam
  webcamBtn.addEventListener('click', loadWebcam);

  // Sample
  sampleBtn.addEventListener('click', loadSample);

  // Drag and drop
  ['dragenter', 'dragover'].forEach((ev) => {
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragging');
    });
  });
  ['dragleave', 'drop'].forEach((ev) => {
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragging');
    });
  });
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) loadFile(f);
  });
  // Click on dropzone (anywhere) → open file picker, except on actual buttons
  dropzone.addEventListener('click', (e) => {
    if (e.target.closest('.btn') || e.target.closest('input')) return;
    fileInput.click();
  });

  // Threshold slider
  thresholdEl.addEventListener('input', () => {
    threshold = parseFloat(thresholdEl.value);
    thresholdDisplay.textContent = threshold.toFixed(2);
  });

  // Play / pause
  playBtn.addEventListener('click', () => {
    if (video.paused) video.play(); else video.pause();
  });
  video.addEventListener('play', () => {
    playIcon.hidden = true;
    pauseIcon.hidden = false;
  });
  video.addEventListener('pause', () => {
    playIcon.hidden = false;
    pauseIcon.hidden = true;
  });

  // Seek bar
  video.addEventListener('timeupdate', updateTime);
  function updateTime() {
    if (!isFinite(video.duration) || video.duration === 0) {
      timeLabel.textContent = formatTime(video.currentTime || 0) + ' / live';
      seek.disabled = true;
      seek.value = 0;
      return;
    }
    seek.disabled = false;
    seek.value = (video.currentTime / video.duration) * 100;
    timeLabel.textContent = `${formatTime(video.currentTime)} / ${formatTime(video.duration)}`;
  }
  seek.addEventListener('input', () => {
    if (!isFinite(video.duration)) return;
    video.currentTime = (seek.value / 100) * video.duration;
  });
  function formatTime(s) {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
  }

  // Reset → back to dropzone
  resetBtn.addEventListener('click', () => {
    cleanupSource();
    dropzone.style.display = '';
    hud.hidden = true;
    controls.hidden = true;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    classStats.clear();
    renderClassList();
  });

  // ---- Init -----------------------------------------------
  // Pre-load model in the background while user picks a video.
  loader.hidden = true;
  bootModel();
})();
