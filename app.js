/* Echtzeit-Objekterkennung — Visual Computing — YOLOv8n · ONNX Runtime Web */

(() => {
  'use strict';

  /* ---- Config ------------------------------------------ */
  const CONF_THRESH = 0.30;
  const IOU_THRESH  = 0.45;
  const INPUT_SIZE  = 640;
  const MODEL_PATH  = 'models/yolov8n.onnx';
  const ORT_CDN     = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';

  /* ---- COCO class labels (80) -------------------------- */
  const CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck',
    'boat','traffic light','fire hydrant','stop sign','parking meter','bench',
    'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',
    'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard',
    'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
    'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl',
    'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
    'donut','cake','chair','couch','potted plant','bed','dining table','toilet',
    'tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
    'toaster','sink','refrigerator','book','clock','vase','scissors',
    'teddy bear','hair drier','toothbrush',
  ];

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
  let session          = null;
  let sessionPromise   = null;
  let rafId            = null;
  let running          = false;
  let currentObjectURL = null;

  // Reuse offscreen canvas every frame to avoid GC pressure
  const offscreen  = document.createElement('canvas');
  offscreen.width  = INPUT_SIZE;
  offscreen.height = INPUT_SIZE;
  const offCtx     = offscreen.getContext('2d', { willReadFrequently: true });

  const fpsWindow      = [];
  const FPS_WINDOW_SIZE = 30;
  let lastFrameTime    = 0;

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

  /* ---- Model loading ----------------------------------- */
  function bootModel() {
    if (sessionPromise) return sessionPromise;
    sessionPromise = (async () => {
      try {
        // Point ORT to its WASM workers on CDN.
        // numThreads=1: GitHub Pages does not send SharedArrayBuffer headers
        // required for multi-threaded WASM.
        ort.env.wasm.wasmPaths  = ORT_CDN;
        ort.env.wasm.numThreads = 1;

        session = await ort.InferenceSession.create(MODEL_PATH, {
          executionProviders:    ['webgl', 'wasm'],
          graphOptimizationLevel: 'all',
        });
        statusDot.classList.add('ready');
        statusText.textContent = 'Bereit · YOLOv8n';
      } catch (err) {
        console.error('ONNX session error:', err);
        statusText.textContent = 'Fehler beim Laden';
        sessionPromise = null;
      }
    })();
    return sessionPromise;
  }

  /* ---- Pre-processing: letterbox → CHW float32 tensor -- */
  function preprocess(videoEl) {
    const vw    = videoEl.videoWidth;
    const vh    = videoEl.videoHeight;
    const scale = Math.min(INPUT_SIZE / vw, INPUT_SIZE / vh);
    const newW  = Math.round(vw * scale);
    const newH  = Math.round(vh * scale);
    const padX  = Math.floor((INPUT_SIZE - newW) / 2);
    const padY  = Math.floor((INPUT_SIZE - newH) / 2);

    // Gray padding preserves the letterbox look expected by YOLOv8
    offCtx.fillStyle = '#808080';
    offCtx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
    offCtx.drawImage(videoEl, padX, padY, newW, newH);

    const { data } = offCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const float32  = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    const area     = INPUT_SIZE * INPUT_SIZE;
    for (let i = 0; i < area; i++) {
      float32[i]          = data[i * 4]     / 255; // R plane
      float32[i + area]   = data[i * 4 + 1] / 255; // G plane
      float32[i + 2*area] = data[i * 4 + 2] / 255; // B plane
    }
    return {
      tensor: new ort.Tensor('float32', float32, [1, 3, INPUT_SIZE, INPUT_SIZE]),
      scale, padX, padY,
    };
  }

  /* ---- Post-processing: decode YOLOv8 output ----------- */
  // YOLOv8 output shape: [1, 84, 8400]
  // Rows 0-3: cx, cy, w, h (in 640×640 pixel space)
  // Rows 4-83: confidence score per COCO class
  function postprocess(raw, scale, padX, padY, origW, origH) {
    const N = 8400;
    const boxes = [], scores = [], classIds = [];

    for (let i = 0; i < N; i++) {
      // Find the class with the highest score
      let maxScore = 0, classId = 0;
      for (let c = 0; c < 80; c++) {
        const s = raw[(4 + c) * N + i];
        if (s > maxScore) { maxScore = s; classId = c; }
      }
      if (maxScore < CONF_THRESH) continue;

      const cx = raw[0 * N + i];
      const cy = raw[1 * N + i];
      const bw = raw[2 * N + i];
      const bh = raw[3 * N + i];

      // Undo letterbox: subtract padding, then divide by scale factor
      const x1 = Math.max(0,     ((cx - bw / 2) - padX) / scale);
      const y1 = Math.max(0,     ((cy - bh / 2) - padY) / scale);
      const x2 = Math.min(origW, ((cx + bw / 2) - padX) / scale);
      const y2 = Math.min(origH, ((cy + bh / 2) - padY) / scale);

      if (x2 <= x1 || y2 <= y1) continue;

      boxes.push([x1, y1, x2, y2]);
      scores.push(maxScore);
      classIds.push(classId);
    }

    return nms(boxes, scores, IOU_THRESH).map((i) => ({
      class: CLASSES[classIds[i]],
      score: scores[i],
      bbox:  [boxes[i][0], boxes[i][1], boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]],
    }));
  }

  /* ---- Non-Maximum Suppression ------------------------- */
  function nms(boxes, scores, iouThr) {
    const order      = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a]);
    const suppressed = new Uint8Array(boxes.length);
    const kept       = [];
    for (const i of order) {
      if (suppressed[i]) continue;
      kept.push(i);
      for (const j of order) {
        if (j === i || suppressed[j]) continue;
        if (iou(boxes[i], boxes[j]) > iouThr) suppressed[j] = 1;
      }
    }
    return kept;
  }

  function iou(a, b) {
    const x1    = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
    const x2    = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const aArea = (a[2] - a[0]) * (a[3] - a[1]);
    const bArea = (b[2] - b[0]) * (b[3] - b[1]);
    return inter / (aArea + bArea - inter + 1e-6);
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
    if (currentObjectURL) { URL.revokeObjectURL(currentObjectURL); currentObjectURL = null; }
    video.srcObject = null;
    video.removeAttribute('src');
    video.load();
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

    dropzone.hidden = true;
    hud.hidden      = false;
    controls.hidden = false;

    try { await video.play(); } catch (_) {}

    if (!session) {
      loader.hidden          = false;
      loaderText.textContent = 'YOLOv8n wird geladen … (12 MB)';
      await bootModel();
      loader.hidden = true;
    }

    startLoop();
    updateTime();
  }

  /* ---- Detection loop ---------------------------------- */
  function startLoop() {
    if (running) return;
    running       = true;
    lastFrameTime = performance.now();
    rafId         = requestAnimationFrame(tick);
  }

  function stopLoop() {
    running = false;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = null;
  }

  async function tick() {
    if (!running) return;

    if (video.readyState >= 2 && !video.paused && !video.ended && session) {
      const { tensor, scale, padX, padY } = preprocess(video);
      let detections = [];
      try {
        const feeds   = { [session.inputNames[0]]: tensor };
        const results = await session.run(feeds);
        const out     = results[session.outputNames[0]];
        detections    = postprocess(out.data, scale, padX, padY, video.videoWidth, video.videoHeight);
        // Free GPU/WASM memory held by output tensors
        for (const t of Object.values(results)) t.dispose?.();
      } catch (e) {
        console.warn('Inference error:', e);
      } finally {
        tensor.dispose?.();
      }

      const now = performance.now();
      const dt  = now - lastFrameTime;
      lastFrameTime = now;
      if (dt > 0 && dt < 1000) {
        fpsWindow.push(1000 / dt);
        if (fpsWindow.length > FPS_WINDOW_SIZE) fpsWindow.shift();
      }
      const fps = fpsWindow.length
        ? fpsWindow.reduce((a, b) => a + b) / fpsWindow.length
        : 0;
      hudFps.textContent = fps.toFixed(1);

      drawOverlay(detections);
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
      timeLabel.textContent = '--:-- / --:--';
      seek.disabled = true;
      return;
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
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    dropzone.hidden = false;
    hud.hidden      = true;
    controls.hidden = true;
  });

  /* ---- Init -------------------------------------------- */
  loader.hidden = true;
  bootModel();
})();
