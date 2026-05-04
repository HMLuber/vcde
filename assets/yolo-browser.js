// OpenCV.js benötigt dieses globale Modul-Objekt, damit es uns benachrichtigt,
// wenn die WebAssembly-Laufzeit von OpenCV vollständig initialisiert ist.
window.Module = {
  onRuntimeInitialized: function () {
    if (typeof window.onOpenCvReady === 'function') {
      window.onOpenCvReady();
    }
  }
};

// ── Konfiguration ────────────────────────────────────────────────────────────
const modelPath      = 'models/yolov5n.onnx';
const inputSize      = 640;
const confThreshold  = 0.20;
const minObjectness  = 0.20;
const minClassScore  = 0.20;
const nmsThreshold   = 0.45;
const modelInputType = 'float16';

// Inferenz-Intervall in ms (höher = weniger CPU-Last, geringere Latenz bei ~100ms)
const INFERENCE_INTERVAL_MS = 100;

const classNames = [
  'person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat',
  'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog',
  'horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella',
  'handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite',
  'baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
  'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich',
  'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa',
  'pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse','remote',
  'keyboard','cell phone','microwave','oven','toaster','sink','refrigerator',
  'book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
];

// ── Globaler Zustand ─────────────────────────────────────────────────────────
let session     = null;
let inputName   = null;
let outputName  = null;
let cvReady     = false;
let ortReady    = false;
let openCvLoaded = false;

// Video-Zustand
let videoElement     = null;
let videoProcessing  = false;
let processingFrame  = false;   // Mutex: verhindert parallele Inferenz-Aufrufe
let lastDetections   = [];      // Zuletzt berechnete Boxen (für Render-Loop)
let renderLoopId     = null;    // requestAnimationFrame Handle
let inferenceTimer   = null;    // setInterval Handle
let frameCount       = 0;
let inferenceCount   = 0;

// ── DOM-Elemente ─────────────────────────────────────────────────────────────
const fileInput     = document.getElementById('fileInput');
const runButton     = document.getElementById('runButton');
const outputCanvas  = document.getElementById('outputCanvas');
const detectionInfo = document.getElementById('detectionInfo');
const debug         = document.getElementById('debug');
const ctx           = outputCanvas.getContext('2d');

// Separates Offscreen-Canvas ausschließlich für Inferenz-Frames
const frameCanvas = document.createElement('canvas');
const frameCtx    = frameCanvas.getContext('2d', { willReadFrequently: true });

// ── Hilfsfunktionen ──────────────────────────────────────────────────────────
function isVideoFile(file) {
  return file && file.type.startsWith('video/');
}

function stopVideoProcessing() {
  videoProcessing = false;

  if (renderLoopId !== null) {
    cancelAnimationFrame(renderLoopId);
    renderLoopId = null;
  }
  if (inferenceTimer !== null) {
    clearInterval(inferenceTimer);
    inferenceTimer = null;
  }
  if (videoElement) {
    videoElement.pause();
    if (videoElement.src) URL.revokeObjectURL(videoElement.src);
    videoElement.remove();
    videoElement = null;
  }

  lastDetections  = [];
  processingFrame = false;
  frameCount      = 0;
  inferenceCount  = 0;
}

function loadVideo(file) {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    video.muted       = true;
    video.playsInline = true;
    video.crossOrigin = 'anonymous';
    video.style.display = 'none';

    video.addEventListener('loadedmetadata', () => {
      if (!video.videoWidth || !video.videoHeight) {
        reject(new Error('Video-Metadaten konnten nicht gelesen werden.'));
        return;
      }
      document.body.appendChild(video);
      resolve(video);
    });

    video.addEventListener('error', () => reject(new Error('Fehler beim Laden des Videos.')));
    video.src = URL.createObjectURL(file);
    video.load();
  });
}

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload  = () => resolve(img);
    img.onerror = () => reject(new Error('Bild konnte nicht geladen werden.'));
    img.src = URL.createObjectURL(file);
  });
}

// ── Modell & OpenCV laden ────────────────────────────────────────────────────
async function loadModel() {
  try {
    console.log('Lade Modell:', modelPath);
    session    = await ort.InferenceSession.create(modelPath, { executionProviders: ['wasm'] });
    inputName  = session.inputNames[0];
    outputName = session.outputNames[0];
    ortReady   = true;
    console.log('Modell geladen. Input:', inputName, '| Output:', outputName);
    loadOpenCv();
    updateReadyState();
  } catch (err) {
    console.error('Modell konnte nicht geladen werden:', err);
    debug.innerText = `Modell-Fehler: ${err.message || err}`;
  }
}

function loadOpenCv() {
  if (openCvLoaded) return;
  openCvLoaded = true;
  const script  = document.createElement('script');
  script.src    = 'https://docs.opencv.org/master/opencv.js';
  script.async  = true;
  script.onerror = () => {
    console.error('OpenCV.js konnte nicht geladen werden.');
    debug.innerText = 'Fehler beim Laden von OpenCV.js';
  };
  document.head.appendChild(script);
}

function updateReadyState() {
  const file = fileInput.files && fileInput.files[0];
  runButton.disabled = !(cvReady && ortReady && file);
}

window.onOpenCvReady = function () {
  cvReady = true;
  console.log('OpenCV bereit.');
  updateReadyState();
};

// ── Preprocessing ────────────────────────────────────────────────────────────
/**
 * Liest ein CanvasImageSource (Image, HTMLCanvasElement, HTMLVideoElement) ein,
 * skaliert es mit Letterbox-Padding auf inputSize × inputSize und gibt die
 * normalisierten Float32-Daten sowie die Skalierungsparameter zurück.
 *
 * WICHTIG: Wir lesen Pixel über ein Hilfs-Canvas (imageDataCanvas), damit
 * cv.imread() immer ein gültiges Canvas-Element bekommt – auch bei direkten
 * HTMLVideoElement-Quellen, bei denen cv.imread() manchmal versagt.
 */
function preprocessSource(source) {
  // Schritt 1: Quelle auf ein temporäres Canvas zeichnen, damit OpenCV es lesen kann
  const w = source.videoWidth  || source.naturalWidth  || source.width;
  const h = source.videoHeight || source.naturalHeight || source.height;

  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width  = w;
  tmpCanvas.height = h;
  tmpCanvas.getContext('2d', { willReadFrequently: true }).drawImage(source, 0, 0, w, h);

  // Schritt 2: OpenCV liest das Canvas und konvertiert RGBA → RGB
  const mat = cv.imread(tmpCanvas);
  const rgb  = new cv.Mat();
  cv.cvtColor(mat, rgb, cv.COLOR_RGBA2RGB);

  // Schritt 3: Letterbox-Skalierung (Seitenverhältnis beibehalten)
  const scale        = Math.min(inputSize / w, inputSize / h);
  const resizedW     = Math.round(w * scale);
  const resizedH     = Math.round(h * scale);
  const padLeft      = Math.floor((inputSize - resizedW) / 2);
  const padTop       = Math.floor((inputSize - resizedH) / 2);
  const padRight     = inputSize - resizedW - padLeft;
  const padBottom    = inputSize - resizedH - padTop;

  const resized = new cv.Mat();
  cv.resize(rgb, resized, new cv.Size(resizedW, resizedH), 0, 0, cv.INTER_LINEAR);

  const padded = new cv.Mat();
  cv.copyMakeBorder(
    resized, padded,
    padTop, padBottom, padLeft, padRight,
    cv.BORDER_CONSTANT, new cv.Scalar(114, 114, 114)
  );

  // Schritt 4: HWC-Byte-Array → CHW-Float32-Array (normalisiert auf [0, 1])
  const floatData = new Float32Array(inputSize * inputSize * 3);
  const pixels    = padded.data;           // Uint8Array, Reihenfolge: R G B R G B ...
  const area      = inputSize * inputSize;

  for (let i = 0; i < area; i++) {
    floatData[i]          = pixels[i * 3]     / 255.0;  // R-Kanal
    floatData[area + i]   = pixels[i * 3 + 1] / 255.0;  // G-Kanal
    floatData[2 * area + i] = pixels[i * 3 + 2] / 255.0; // B-Kanal
  }

  // OpenCV-Speicher freigeben (verhindert Memory-Leaks in WASM)
  mat.delete();
  rgb.delete();
  resized.delete();
  padded.delete();

  return { floatData, scale, padLeft, padTop, srcWidth: w, srcHeight: h };
}

// ── Float32 → Float16 Konvertierung ─────────────────────────────────────────
function float32ToFloat16(value) {
  const f32 = new Float32Array(1);
  const i32 = new Int32Array(f32.buffer);
  f32[0] = value;
  const x        = i32[0];
  const sign     = (x >> 16) & 0x8000;
  const rawExp   = (x >> 23) & 0xff;
  let   exp      = rawExp - 127 + 15;
  let   mantissa = x & 0x007fffff;

  if (rawExp === 255) return sign | 0x7c00 | (mantissa ? 0x200 : 0);
  if (exp <= 0) {
    if (exp < -10) return sign;
    mantissa = (mantissa | 0x00800000) >> (1 - exp);
    return sign | ((mantissa + 0x0fff + ((mantissa >> 13) & 1)) >> 13);
  }
  if (exp > 30) return sign | 0x7c00;
  return sign | (exp << 10) | (mantissa >> 13);
}

function toFloat16Array(float32Array) {
  const out = new Uint16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    out[i] = float32ToFloat16(float32Array[i]);
  }
  return out;
}

// ── Non-Maximum-Suppression ──────────────────────────────────────────────────
function nonMaxSuppression(boxes, scores, threshold) {
  const order = scores
    .map((score, idx) => ({ score, idx }))
    .sort((a, b) => b.score - a.score)
    .map(item => item.idx);

  const keep = [];
  while (order.length > 0) {
    const idx  = order.shift();
    keep.push(idx);
    const [xA1, yA1, wA, hA] = boxes[idx];
    const xA2 = xA1 + wA, yA2 = yA1 + hA;

    for (let j = order.length - 1; j >= 0; j--) {
      const [xB1, yB1, wB, hB] = boxes[order[j]];
      const xB2 = xB1 + wB, yB2 = yB1 + hB;

      const interW  = Math.max(0, Math.min(xA2, xB2) - Math.max(xA1, xB1));
      const interH  = Math.max(0, Math.min(yA2, yB2) - Math.max(yA1, yB1));
      const inter   = interW * interH;
      const union   = wA * hA + wB * hB - inter;
      if (inter / union > threshold) order.splice(j, 1);
    }
  }
  return keep;
}

// ── Inferenz-Kern ────────────────────────────────────────────────────────────
/**
 * Führt Preprocessing, ONNX-Inferenz und NMS auf einer beliebigen
 * CanvasImageSource durch. Gibt das Array der gefilterten Detektionen zurück.
 *
 * Die Koordinaten werden direkt auf die Pixelgröße der Quelle (srcWidth × srcHeight)
 * zurückgerechnet, damit sie im Render-Loop korrekt skaliert werden können.
 */
async function runInference(source) {
  const { floatData, scale, padLeft, padTop, srcWidth, srcHeight } =
    preprocessSource(source);

  const inputData   = modelInputType === 'float16'
    ? toFloat16Array(floatData)
    : floatData;
  const inputTensor = new ort.Tensor(modelInputType, inputData, [1, 3, inputSize, inputSize]);
  const results     = await session.run({ [inputName]: inputTensor });
  const output      = results[outputName].data;

  const raw          = [];
  const numDetections = output.length / 85;

  for (let i = 0; i < numDetections; i++) {
    const base       = i * 85;
    const objectness = output[base + 4];
    if (objectness < minObjectness) continue;

    let classId = 0, classScore = 0;
    for (let c = 0; c < 80; c++) {
      const s = output[base + 5 + c];
      if (s > classScore) { classScore = s; classId = c; }
    }
    if (classScore < minClassScore) continue;

    const confidence = objectness * classScore;
    if (confidence < confThreshold) continue;

    // YOLO gibt Mittelpunkt + Breite/Höhe zurück (absolut in inputSize-Koordinaten)
    const cx = output[base], cy = output[base + 1];
    const bw = output[base + 2], bh = output[base + 3];

    // Normalisiert (0..1) oder absolut (0..640) – beides abfangen
    const isNorm = cx <= 1 && cy <= 1 && bw <= 1 && bh <= 1;
    const px = isNorm ? cx * inputSize : cx;
    const py = isNorm ? cy * inputSize : cy;
    const pw = isNorm ? bw * inputSize : bw;
    const ph = isNorm ? bh * inputSize : bh;

    // Letterbox rückgängig machen → Quellbild-Koordinaten
    const x = (px - pw / 2 - padLeft) / scale;
    const y = (py - ph / 2 - padTop)  / scale;
    const w = pw / scale;
    const h = ph / scale;

    // Auf Bildgrenzen begrenzen
    const cx1 = Math.max(0, Math.min(srcWidth,  x));
    const cy1 = Math.max(0, Math.min(srcHeight, y));
    const cw  = Math.max(0, Math.min(srcWidth  - cx1, w));
    const ch  = Math.max(0, Math.min(srcHeight - cy1, h));

    raw.push({ classId, classScore, objectness, confidence, box: [cx1, cy1, cw, ch] });
  }

  const boxes  = raw.map(d => d.box);
  const scores = raw.map(d => d.confidence);
  const keep   = nonMaxSuppression(boxes, scores, nmsThreshold);
  return keep.map(i => raw[i]);
}

// ── Zeichnen ─────────────────────────────────────────────────────────────────
/**
 * Zeichnet eine Bounding-Box mit Label auf das outputCanvas.
 * scaleX / scaleY ermöglichen die Skalierung von Quellkoordinaten auf Canvas-Koordinaten.
 */
function drawBox(box, label, color, scaleX = 1, scaleY = 1) {
  const [bx, by, bw, bh] = [
    box[0] * scaleX,
    box[1] * scaleY,
    box[2] * scaleX,
    box[3] * scaleY
  ];

  ctx.lineWidth   = 2;
  ctx.strokeStyle = color;
  ctx.strokeRect(bx, by, bw, bh);

  ctx.font = '16px Arial';
  const tw = ctx.measureText(label).width + 8;
  const th = 20;
  ctx.fillStyle = color;
  ctx.fillRect(bx, by - th - 2, tw, th);
  ctx.fillStyle = '#000';
  ctx.fillText(label, bx + 4, by - 6);
}

// ── Render-Loop (requestAnimationFrame) ──────────────────────────────────────
/**
 * Zeichnet jeden Frame: erst das aktuelle Video-Frame, dann alle gespeicherten
 * Detektionen. Koordinaten wurden in Quellbildgröße gespeichert → Skalierung
 * auf Canvas-Größe erfolgt hier über scaleX / scaleY.
 */
function renderLoop() {
  if (!videoProcessing) return;
  if (videoElement.paused || videoElement.ended) {
    if (videoElement.ended) {
      detectionInfo.innerHTML += ' — Videoende erreicht.';
      stopVideoProcessing();
    }
    return;
  }

  // Video-Frame auf Canvas zeichnen
  ctx.drawImage(videoElement, 0, 0, outputCanvas.width, outputCanvas.height);

  // Skalierungsfaktoren: Quellbild → Canvas
  const scaleX = outputCanvas.width  / videoElement.videoWidth;
  const scaleY = outputCanvas.height / videoElement.videoHeight;

  // Detektionen overlay
  lastDetections.forEach(det => {
    const name  = classNames[det.classId] || 'unknown';
    const label = `${name} ${(det.confidence * 100).toFixed(0)}%`;
    drawBox(det.box, label, 'magenta', scaleX, scaleY);
  });

  detectionInfo.innerHTML =
    `Frame: ${frameCount} | Inferenz-Runs: ${inferenceCount} | ` +
    `Erkennungen: ${lastDetections.length}`;

  frameCount++;
  renderLoopId = requestAnimationFrame(renderLoop);
}

// ── Inferenz-Loop (setInterval) ───────────────────────────────────────────────
/**
 * Läuft asynchron parallel zum Render-Loop. Kopiert den aktuellen Video-Frame
 * auf das Offscreen-Canvas und führt dann ONNX-Inferenz durch.
 * Der Mutex `processingFrame` verhindert überlappende Inferenz-Aufrufe.
 */
function startInferenceLoop() {
  inferenceTimer = setInterval(async () => {
    if (!videoProcessing || processingFrame || !videoElement || videoElement.paused) return;

    processingFrame = true;
    try {
      // Frame in Offscreen-Canvas kopieren (konsistenter Snapshot für Inferenz)
      frameCanvas.width  = videoElement.videoWidth;
      frameCanvas.height = videoElement.videoHeight;
      frameCtx.drawImage(videoElement, 0, 0);

      // Inferenz auf Offscreen-Canvas (nicht auf videoElement direkt, da cv.imread
      // bei HTMLVideoElement unzuverlässig ist)
      lastDetections = await runInference(frameCanvas);
      inferenceCount++;
    } catch (err) {
      console.error('Inferenz-Fehler:', err);
      debug.innerText = `Inferenz-Fehler: ${err.message || err}`;
    } finally {
      processingFrame = false;
    }
  }, INFERENCE_INTERVAL_MS);
}

// ── Haupt-Einstiegspunkt ─────────────────────────────────────────────────────
async function runDetection() {
  if (!fileInput.files || !fileInput.files[0]) return;
  const file = fileInput.files[0];

  // Vorherige Verarbeitung stoppen
  stopVideoProcessing();

  runButton.disabled = true;
  detectionInfo.innerHTML = 'Starte Erkennung…';

  try {
    if (isVideoFile(file)) {
      // ── Video-Modus ───────────────────────────────────────────────────────
      videoElement = await loadVideo(file);

      // Canvas auf Video-Auflösung setzen (max. 1280 px Breite für Performance)
      const maxW = 1280;
      const vw   = videoElement.videoWidth;
      const vh   = videoElement.videoHeight;
      if (vw > maxW) {
        outputCanvas.width  = maxW;
        outputCanvas.height = Math.round(vh * maxW / vw);
      } else {
        outputCanvas.width  = vw;
        outputCanvas.height = vh;
      }

      videoProcessing = true;
      await videoElement.play();

      renderLoop();           // Render-Loop starten
      startInferenceLoop();   // Inferenz-Loop starten

    } else {
      // ── Bild-Modus ────────────────────────────────────────────────────────
      const img = await loadImage(file);
      outputCanvas.width  = img.naturalWidth;
      outputCanvas.height = img.naturalHeight;

      ctx.drawImage(img, 0, 0);
      detectionInfo.innerHTML = 'Inferenz läuft…';

      const detections = await runInference(img);

      ctx.drawImage(img, 0, 0);   // Canvas frisch zeichnen vor den Boxen
      if (detections.length === 0) {
        detectionInfo.innerHTML = 'Keine Objekte erkannt.';
      } else {
        detections.forEach(det => {
          const name  = classNames[det.classId] || 'unknown';
          const label = `${name} ${(det.confidence * 100).toFixed(0)}%`;
          drawBox(det.box, label, 'magenta');
        });
        detectionInfo.innerHTML =
          `Erkennung abgeschlossen — ${detections.length} Objekt(e) gefunden.`;
      }
      runButton.disabled = false;
    }
  } catch (err) {
    console.error('Erkennungsfehler:', err);
    detectionInfo.innerHTML = `Fehler: ${err.message || err}`;
    if (err.stack) debug.innerText = err.stack;
    runButton.disabled = false;
  }
}

// ── Event-Listener ───────────────────────────────────────────────────────────
fileInput.addEventListener('change', () => {
  stopVideoProcessing();
  const file = fileInput.files && fileInput.files[0];
  if (file) {
    detectionInfo.innerHTML = isVideoFile(file)
      ? 'Video ausgewählt. Drücke „Erkennung starten".'
      : 'Bild ausgewählt. Drücke „Erkennung starten".';
  } else {
    detectionInfo.innerHTML = 'Warte auf Eingabe…';
  }
  updateReadyState();
});

runButton.addEventListener('click', runDetection);

window.addEventListener('error', (e) => {
  console.error('Globaler Fehler:', e.error || e.message);
  debug.innerText = `Fehler: ${e.error?.message || e.message}`;
});

window.addEventListener('unhandledrejection', (e) => {
  console.error('Unbehandelte Promise-Ablehnung:', e.reason);
  debug.innerText = `Promise-Fehler: ${e.reason?.message || e.reason}`;
});

// ── Start ─────────────────────────────────────────────────────────────────────
loadModel();