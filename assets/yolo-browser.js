window.Module = {
  onRuntimeInitialized: function() {
    if (typeof window.onOpenCvReady === 'function') {
      window.onOpenCvReady();
    }
  }
};

const modelPath = 'models/yolov5n.onnx';
const inputSize = 640;
const confThreshold = 0.30;
const minObjectness = 0.30;
const minClassScore = 0.35;
const nmsThreshold = 0.45;
const colorMap = {
  person: 'lime',
  tie: 'yellow',
  cat: 'red',
  dog: 'cyan'
};
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

let session = null;
let inputName = null;
let outputName = null;
let cvReady = false;
let ortReady = false;
let openCvLoaded = false;

const statusEl = document.getElementById('status');
const fileInput = document.getElementById('fileInput');
const runButton = document.getElementById('runButton');
const outputCanvas = document.getElementById('outputCanvas');
const detectionInfo = document.getElementById('detectionInfo');
const debug = document.getElementById('debug');
const ctx = outputCanvas.getContext('2d');

async function loadModel() {
  try {
    console.log('Versuche Modell zu laden:', modelPath);
    statusEl.innerText = 'Lade YOLO-Modell...';
    session = await ort.InferenceSession.create(modelPath, { executionProviders: ['wasm'] });
    inputName = session.inputNames[0];
    outputName = session.outputNames[0];
    ortReady = true;
    console.log('Modell geladen:', modelPath);
    statusEl.innerText = 'Modell geladen. Lade OpenCV...';
    loadOpenCv();
    updateReadyState();
  } catch (err) {
    console.error('Modell konnte nicht geladen werden.', err);
    statusEl.innerText = 'Modell konnte nicht geladen werden. Siehe Konsole.';
    debug.innerText = `Modell-Fehler: ${err.message || err.toString()}`;
    if (err.stack) debug.innerText += `\n${err.stack}`;
  }
}

function loadOpenCv() {
  if (openCvLoaded) return;
  openCvLoaded = true;
  statusEl.innerText = 'Lade OpenCV...';
  const script = document.createElement('script');
  script.src = 'https://docs.opencv.org/master/opencv.js';
  script.async = true;
  script.onerror = () => {
    console.error('OpenCV konnte nicht geladen werden.');
    statusEl.innerText = 'OpenCV konnte nicht geladen werden. Siehe Konsole.';
    debug.innerText = 'Fehler beim Laden von OpenCV.js';
  };
  document.head.appendChild(script);
}

function updateReadyState() {
  if (cvReady && ortReady) {
    runButton.disabled = false;
    statusEl.innerText = 'Bereit. Wähle ein Bild und starte die Erkennung.';
  }
}

window.onOpenCvReady = function() {
  cvReady = true;
  console.log('OpenCV geladen.');
  statusEl.innerText = ortReady ? 'Bereit. Wähle ein Bild.' : 'OpenCV geladen. Lade YOLO-Modell...';
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
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

function preprocessImage(img) {
  const mat = cv.imread(img);
  const rgb = new cv.Mat();
  cv.cvtColor(mat, rgb, cv.COLOR_RGBA2RGB);
  const resized = new cv.Mat();
  const dsize = new cv.Size(inputSize, inputSize);
  cv.resize(rgb, resized, dsize, 0, 0, cv.INTER_AREA);

  const floatData = new Float32Array(inputSize * inputSize * 3);
  for (let y = 0; y < inputSize; y++) {
    for (let x = 0; x < inputSize; x++) {
      const i = y * inputSize + x;
      floatData[i] = resized.data[i * 3] / 255.0;
      floatData[inputSize * inputSize + i] = resized.data[i * 3 + 1] / 255.0;
      floatData[2 * inputSize * inputSize + i] = resized.data[i * 3 + 2] / 255.0;
    }
  }

  mat.delete();
  rgb.delete();
  resized.delete();
  return floatData;
}

function nonMaxSuppression(boxes, scores, threshold) {
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

async function runDetection() {
  if (!fileInput.files || !fileInput.files[0]) return;
  const img = await loadImage(fileInput.files[0]);
  outputCanvas.width = img.naturalWidth;
  outputCanvas.height = img.naturalHeight;
  ctx.drawImage(img, 0, 0, outputCanvas.width, outputCanvas.height);

  statusEl.innerText = 'Verarbeite Bild...';
  const floatData = preprocessImage(img);
  const inputTensor = new ort.Tensor('float32', floatData, [1, 3, inputSize, inputSize]);
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
    const x = (cx - w / 2) * outputCanvas.width;
    const y = (cy - h / 2) * outputCanvas.height;
    const width = w * outputCanvas.width;
    const height = h * outputCanvas.height;
    detections.push({ classId, classScore, objectness, confidence, box: [x, y, width, height] });
  }

  const boxes = detections.map(d => d.box);
  const scores = detections.map(d => d.confidence);
  const keep = nonMaxSuppression(boxes, scores, nmsThreshold);
  const selected = keep.map(i => detections[i]);

  ctx.drawImage(img, 0, 0, outputCanvas.width, outputCanvas.height);
  if (selected.length === 0) {
    detectionInfo.innerHTML = 'Keine Objekte erkannt.';
  } else {
    detectionInfo.innerHTML = `<strong>Erkannte Objekte:</strong> ${selected.length}`;
    const list = document.createElement('ul');
    selected.forEach(det => {
      const name = classNames[det.classId] || 'unknown';
      const label = `${name} (${det.confidence.toFixed(2)})`;
      const li = document.createElement('li');
      li.innerText = label;
      list.appendChild(li);
      drawBox(det.box, label, colorMap[name] || 'magenta');
    });
    detectionInfo.appendChild(list);
  }
  statusEl.innerText = 'Erkennung abgeschlossen.';
}

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
  runButton.disabled = !fileInput.files.length || !cvReady || !ortReady;
  detectionInfo.innerHTML = 'Bereit zur Erkennung.';
});

runButton.addEventListener('click', runDetection);
loadModel();
