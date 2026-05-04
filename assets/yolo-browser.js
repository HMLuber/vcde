// OpenCV.js benötigt dieses globale Modul-Objekt, damit es uns benachrichtigt,
// wenn die WebAssembly-Laufzeit von OpenCV vollständig initialisiert ist.
window.Module = {
  onRuntimeInitialized: function() {
    if (typeof window.onOpenCvReady === 'function') {
      window.onOpenCvReady();
    }
  }
};

// Pfad zum ONNX-Modell, das im Browser geladen und ausgeführt wird.
const modelPath = 'models/yolov5n.onnx';
// Das Eintragsbild wird auf 640x640 pixel verarbeitet, wie das YOLO-Modell erwartet.
const inputSize = 640;
// Schwellwerte für die Erkennung: je niedriger, desto empfindlicher die Erkennung.
const confThreshold = 0.20;
const minObjectness = 0.20;
const minClassScore = 0.20;
const nmsThreshold = 0.45;
const classNames = [
  'person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat',
  'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog',
  'horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella',
  'handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat',
  'baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork',
  'knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog',
  'pizza','donut','cake','chair','sofa','pottedplant','bed','diningtable','toilet','tvmonitor',
  'laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator',
  'book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
];

// Das Modell erwartet float16-Eingaben. Wir konvertieren die Pixelwerte daher
// von Float32 nach Float16, bevor sie als Tensor an den ONNX-Laufzeit übergeben werden.
const modelInputType = 'float16';

// Globale Variablen für das ONNX-Session-Objekt und den Zustand der Bibliotheken.
let session = null;
let inputName = null;
let outputName = null;
let cvReady = false;
let ortReady = false;
let openCvLoaded = false;

// HTML-Elemente aus der Seite, die wir später aktualisieren und lesen.
const fileInput = document.getElementById('fileInput');
const runButton = document.getElementById('runButton');
const outputCanvas = document.getElementById('outputCanvas');
const detectionInfo = document.getElementById('detectionInfo');
const debug = document.getElementById('debug');
const ctx = outputCanvas.getContext('2d');
const frameCanvas = document.createElement('canvas');
const frameCtx = frameCanvas.getContext('2d');
let currentMediaType = null;
let videoElement = null;
let videoProcessing = false;
let processingVideoFrame = false;
let pendingVideoFrame = false;

function isVideoFile(file) {
  return file && file.type.startsWith('video/');
}

function loadVideo(file) {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    video.muted = true;
    video.playsInline = true;
    video.crossOrigin = 'anonymous';
    video.style.display = 'none';
    video.src = URL.createObjectURL(file);

    video.addEventListener('loadedmetadata', () => {
      if (!video.videoWidth || !video.videoHeight) {
        reject(new Error('Video konnte nicht geladen werden.'));
        return;
      }
      document.body.appendChild(video);
      resolve(video);
    });

    video.addEventListener('error', (event) => {
      reject(new Error('Fehler beim Laden des Videos.'));
    });

    video.load();
  });
}

function stopVideoProcessing() {
  videoProcessing = false;
  if (videoElement) {
    videoElement.pause();
    URL.revokeObjectURL(videoElement.src);
    videoElement.remove();
    videoElement = null;
  }
}

async function loadModel() {
  // Lädt das ONNX-Modell asynchron in den Browser. Die ONNX Runtime kann dann
  // das Modell direkt in WebAssembly ausführen.
  try {
    console.log('Versuche Modell zu laden:', modelPath);
    session = await ort.InferenceSession.create(modelPath, { executionProviders: ['wasm'] });
    // Merke uns die Namen der Eingabe- und Ausgabe-Tensoren für später.
    inputName = session.inputNames[0];
    outputName = session.outputNames[0];
    ortReady = true;
    console.log('Modell geladen:', modelPath);
    console.log('Session inputNames:', session.inputNames);
    console.log('Session outputNames:', session.outputNames);
    console.log('Verwendeter Eingabetyp:', modelInputType);
    loadOpenCv();
    updateReadyState();
  } catch (err) {
    console.error('Modell konnte nicht geladen werden.', err);
    debug.innerText = `Modell-Fehler: ${err.message || err.toString()}`;
    if (err.stack) debug.innerText += `\n${err.stack}`;
  }
}

function loadOpenCv() {
  // Lädt OpenCV.js dynamisch als <script>-Tag. OpenCV.js wird benötigt, um das
  // Eingangsbild zu lesen, zu skalieren und in das YOLO-Format zu bringen.
  if (openCvLoaded) return;
  openCvLoaded = true;
  const script = document.createElement('script');
  script.src = 'https://docs.opencv.org/master/opencv.js';
  script.async = true;
  script.onerror = () => {
    console.error('OpenCV konnte nicht geladen werden.');
    debug.innerText = 'Fehler beim Laden von OpenCV.js';
  };
  document.head.appendChild(script);
}

function updateReadyState() {
  if (cvReady && ortReady) {
    runButton.disabled = false;
  }
}

window.onOpenCvReady = function() {
  cvReady = true;
  console.log('OpenCV geladen.');
  updateReadyState();
};

window.addEventListener('error', (event) => {
  console.error('Unerwarteter Fehler:', event.error || event.message);
  debug.innerText = `Fehler: ${event.error?.message || event.message}`;
});

window.addEventListener('unhandledrejection', (event) => {
  console.error('Unhandled Promise Rejection:', event.reason);
  debug.innerText = `Promise-Fehler: ${event.reason?.message || event.reason}`;
});

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = (event) => reject(new Error('Bild konnte nicht geladen werden.'));
    img.src = URL.createObjectURL(file);
  });
}

function drawVideoFrame(video) {
  frameCanvas.width = video.videoWidth;
  frameCanvas.height = video.videoHeight;
  frameCtx.drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);
  return frameCanvas;
}

async function runDetectionOnSource(source, frameLabel = '') {
  const { floatData, scale, padLeft, padTop } = preprocessImage(source);
  const inputData = modelInputType === 'float16' ? convertFloat32ToFloat16Array(floatData) : floatData;
  const inputTensor = new ort.Tensor(modelInputType, inputData, [1, 3, inputSize, inputSize]);
  const feeds = { [inputName]: inputTensor };
  const results = await session.run(feeds);
  const outputData = results[outputName].data;

  const detections = [];
  const numDetections = outputData.length / 85;

  for (let i = 0; i < numDetections; i++) {
    const base = i * 85;
    const objectness = outputData[base + 4];
    let classId = 0;
    let classScore = 0;
    for (let c = 0; c < 80; c++) {
      const score = outputData[base + 5 + c];
      if (score > classScore) {
        classScore = score;
        classId = c;
      }
    }
    const confidence = objectness * classScore;
    if (objectness < minObjectness || classScore < minClassScore || confidence < confThreshold) continue;
    const cx = outputData[base];
    const cy = outputData[base + 1];
    const w = outputData[base + 2];
    const h = outputData[base + 3];
    let x;
    let y;
    let width;
    let height;
    const normalized = cx <= 1 && cy <= 1 && w <= 1 && h <= 1;
    if (normalized) {
      x = ((cx * inputSize) - (w * inputSize) / 2 - padLeft) / scale;
      y = ((cy * inputSize) - (h * inputSize) / 2 - padTop) / scale;
      width = (w * inputSize) / scale;
      height = (h * inputSize) / scale;
    } else {
      x = (cx - w / 2 - padLeft) / scale;
      y = (cy - h / 2 - padTop) / scale;
      width = w / scale;
      height = h / scale;
      console.log('Detected coordinates appear absolute, switching mapping:', { cx, cy, w, h, normalized });
    }
    const clampedX = Math.max(0, Math.min(outputCanvas.width, x));
    const clampedY = Math.max(0, Math.min(outputCanvas.height, y));
    const clampedW = Math.max(0, Math.min(outputCanvas.width - clampedX, width));
    const clampedH = Math.max(0, Math.min(outputCanvas.height - clampedY, height));
    detections.push({ classId, classScore, objectness, confidence, box: [clampedX, clampedY, clampedW, clampedH] });
  }

  const boxes = detections.map(d => d.box);
  const scores = detections.map(d => d.confidence);
  const keep = nonMaxSuppression(boxes, scores, nmsThreshold);
  const selected = keep.map(i => detections[i]);

  ctx.drawImage(source, 0, 0, outputCanvas.width, outputCanvas.height);
  const infoPrefix = frameLabel ? `${frameLabel} ` : '';
  if (selected.length === 0) {
    detectionInfo.innerHTML = `${infoPrefix}Keine Objekte erkannt.`;
  } else {
    detectionInfo.innerHTML = `${infoPrefix}<strong>Erkannte Objekte:</strong> ${selected.length}`;
    selected.forEach(det => {
      const name = classNames[det.classId] || 'unknown';
      const label = `${name} (${det.confidence.toFixed(2)})`;
      drawBox(det.box, label, 'magenta');
    });
  }
}

function preprocessImage(img) {
  const mat = cv.imread(img);
  const rgb = new cv.Mat();
  cv.cvtColor(mat, rgb, cv.COLOR_RGBA2RGB);

  const srcHeight = rgb.rows;
  const srcWidth = rgb.cols;
  const scale = Math.min(inputSize / srcWidth, inputSize / srcHeight);
  const resizedWidth = Math.round(srcWidth * scale);
  const resizedHeight = Math.round(srcHeight * scale);

  const resized = new cv.Mat();
  cv.resize(rgb, resized, new cv.Size(resizedWidth, resizedHeight), 0, 0, cv.INTER_AREA);

  const top = Math.floor((inputSize - resizedHeight) / 2);
  const bottom = inputSize - resizedHeight - top;
  const left = Math.floor((inputSize - resizedWidth) / 2);
  const right = inputSize - resizedWidth - left;

  const padded = new cv.Mat();
  cv.copyMakeBorder(resized, padded, top, bottom, left, right, cv.BORDER_CONSTANT, new cv.Scalar(114, 114, 114));

  const floatData = new Float32Array(inputSize * inputSize * 3);
  for (let y = 0; y < inputSize; y++) {
    for (let x = 0; x < inputSize; x++) {
      const i = y * inputSize + x;
      floatData[i] = padded.data[i * 3] / 255.0;
      floatData[inputSize * inputSize + i] = padded.data[i * 3 + 1] / 255.0;
      floatData[2 * inputSize * inputSize + i] = padded.data[i * 3 + 2] / 255.0;
    }
  }

  mat.delete();
  rgb.delete();
  resized.delete();
  padded.delete();
  return { floatData, scale, padLeft: left, padTop: top };
}

// ONNX Runtime Web unterstützt float16-Eingaben, aber JavaScript hat keinen
// nativen Float16-Typ. Deshalb wandeln wir jeden Float32-Wert selbst um.
function float32ToFloat16(value) {
  const floatView = new Float32Array(1);
  const int32View = new Int32Array(floatView.buffer);
  floatView[0] = value;

  // Diese Bitoperationen konvertieren die 32-Bit-Fließkommazahl in das 16-Bit-Format.
  const x = int32View[0];
  const sign = (x >> 16) & 0x8000;
  const rawExponent = (x >> 23) & 0xff;
  let exponent = rawExponent - 127 + 15;
  let mantissa = x & 0x007fffff;

  if (rawExponent === 255) {
    return sign | 0x7c00 | (mantissa ? 0x200 : 0);
  }

  if (exponent <= 0) {
    if (exponent < -10) {
      return sign;
    }
    mantissa = (mantissa | 0x00800000) >> (1 - exponent);
    return sign | ((mantissa + 0x0fff + ((mantissa >> 13) & 1)) >> 13);
  }

  if (exponent > 30) {
    return sign | 0x7c00;
  }

  return sign | (exponent << 10) | (mantissa >> 13);
}

function convertFloat32ToFloat16Array(float32Array) {
  // Konvertiert das gesamte Bilder-Array von Float32 nach Float16.
  const float16Array = new Uint16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    float16Array[i] = float32ToFloat16(float32Array[i]);
  }
  return float16Array;
}

function nonMaxSuppression(boxes, scores, threshold) {
  // Diese Funktion entfernt sich stark überlappende Boxen, damit nur die beste
  // Box pro Objekt erhalten bleibt. Das ist wichtig, weil YOLO viele ähnliche
  // Detektionen für ein Objekt erzeugen kann.
  const order = scores.map((score, idx) => ({ score, idx })).sort((a, b) => b.score - a.score).map(item => item.idx);
  const keep = [];
  while (order.length > 0) {
    const idx = order.shift();
    keep.push(idx);
    const boxA = boxes[idx];
    const [xA1, yA1, xA2, yA2] = [boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]];
    for (let j = order.length - 1; j >= 0; j--) {
      const boxB = boxes[order[j]];
      const [xB1, yB1, xB2, yB2] = [boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]];
      const interArea = Math.max(0, Math.min(xA2, xB2) - Math.max(xA1, xB1)) * Math.max(0, Math.min(yA2, yB2) - Math.max(yA1, yB1));
      const areaA = (xA2 - xA1) * (yA2 - yA1);
      const areaB = (xB2 - xB1) * (yB2 - yB1);
      const ovr = interArea / (areaA + areaB - interArea);
      if (ovr > threshold) order.splice(j, 1);
    }
  }
  return keep;
}

// Hauptfunktion: wird aufgerufen, wenn der Benutzer ein Bild hochlädt und die Erkennung startet.
// Sie führt alle Schritte aus: Laden, Preprocessing, Inferenz, NMS und Zeichnen.
async function runDetection() {
  if (!fileInput.files || !fileInput.files[0]) return;
  const file = fileInput.files[0];
  if (videoProcessing) {
    stopVideoProcessing();
  }

  try {
    if (isVideoFile(file)) {
      videoElement = await loadVideo(file);
      outputCanvas.width = videoElement.videoWidth;
      outputCanvas.height = videoElement.videoHeight;
      detectionInfo.innerHTML = 'Video geladen. Erkennung startet...';
      videoProcessing = true;

      await videoElement.play();

      const processVideoFrame = async () => {
        if (!videoProcessing || videoElement.paused || videoElement.ended) {
          videoProcessing = false;
          if (videoElement && videoElement.ended) {
            detectionInfo.innerHTML += ' (Videoende erreicht)';
          }
          return;
        }

        if (processingVideoFrame) {
          pendingVideoFrame = true;
          return;
        }

        processingVideoFrame = true;
        try {
          const frame = drawVideoFrame(videoElement);
          const frameLabel = `Zeit ${videoElement.currentTime.toFixed(2)}s`;
          await runDetectionOnSource(frame, frameLabel);
        } catch (err) {
          console.error('Fehler bei der Frame-Verarbeitung.', err);
        } finally {
          processingVideoFrame = false;
        }

        if (pendingVideoFrame) {
          pendingVideoFrame = false;
          processVideoFrame();
          return;
        }

        if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
          videoElement.requestVideoFrameCallback(() => processVideoFrame());
        } else {
          requestAnimationFrame(processVideoFrame);
        }
      };

      processVideoFrame();
    } else {
      const img = await loadImage(file);
      outputCanvas.width = img.naturalWidth;
      outputCanvas.height = img.naturalHeight;
      await runDetectionOnSource(img);
    }
  } catch (err) {
    console.error('Fehler bei der Erkennung.', err);
    detectionInfo.innerText = `Erkennungsfehler: ${err.message || err.toString()}`;
  }
}

// Zeichnet eine erkannte Box und das zugehörige Label auf das Canvas.
function drawBox(box, label, color) {
  ctx.lineWidth = 3;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.strokeRect(box[0], box[1], box[2], box[3]);
  ctx.font = '18px Arial';
  ctx.fillStyle = color;
  const textWidth = ctx.measureText(label).width + 8;
  const textHeight = 22;
  ctx.fillRect(box[0], box[1] - textHeight - 4, textWidth, textHeight);
  ctx.fillStyle = '#000';
  ctx.fillText(label, box[0] + 4, box[1] - 8);
}

fileInput.addEventListener('change', () => {
  const file = fileInput.files && fileInput.files[0];
  if (videoProcessing) {
    stopVideoProcessing();
  }
  currentMediaType = file ? (isVideoFile(file) ? 'video' : 'image') : null;
  runButton.disabled = !file || !cvReady || !ortReady;
  if (!file) {
    detectionInfo.innerHTML = 'Warte auf Eingabe...';
  } else if (currentMediaType === 'video') {
    detectionInfo.innerHTML = 'Video geladen. Drücke Erkennung starten.';
  } else {
    detectionInfo.innerHTML = 'Bereit zur Erkennung.';
  }
});

runButton.addEventListener('click', runDetection);
loadModel();
