/* YOLOv8n inference worker — runs in a separate thread so the UI never blocks */
'use strict';

importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js');

const CONF_THRESH = 0.30;
const IOU_THRESH  = 0.45;
const INPUT_SIZE  = 320;
// N_DET = (S/8)² + (S/16)² + (S/32)²  for input size S
// 320px → 40²+20²+10² = 1600+400+100 = 2100
const N_DET       = 2100;
const ORT_CDN     = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';

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

let session = null;

/* ---- Model loading ---------------------------------------- */
async function loadModel() {
  try {
    ort.env.wasm.wasmPaths  = ORT_CDN;
    ort.env.wasm.numThreads = 1; // GitHub Pages blocks SharedArrayBuffer
    session = await ort.InferenceSession.create('models/yolov8n.onnx', {
      executionProviders:     ['wasm'],
      graphOptimizationLevel: 'all',
    });
    self.postMessage({ type: 'ready', provider: 'WASM' });
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err) });
  }
}

/* ---- Post-processing: decode raw [1, 84, 2100] output ----- */
function postprocess(raw, scale, padX, padY, origW, origH) {
  const boxes = [], scores = [], classIds = [];

  for (let i = 0; i < N_DET; i++) {
    let maxScore = 0, classId = 0;
    for (let c = 0; c < 80; c++) {
      const s = raw[(4 + c) * N_DET + i];
      if (s > maxScore) { maxScore = s; classId = c; }
    }
    if (maxScore < CONF_THRESH) continue;

    const cx = raw[0 * N_DET + i];
    const cy = raw[1 * N_DET + i];
    const bw = raw[2 * N_DET + i];
    const bh = raw[3 * N_DET + i];

    // Undo letterbox: subtract padding, divide by scale
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

/* ---- NMS -------------------------------------------------- */
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
  return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-6);
}

/* ---- Message handler -------------------------------------- */
self.onmessage = async (e) => {
  if (e.data.type !== 'infer' || !session) return;

  const { float32, scale, padX, padY, origW, origH } = e.data;
  const tensor = new ort.Tensor('float32', float32, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  let detections = [];
  try {
    const results = await session.run({ [session.inputNames[0]]: tensor });
    const out     = results[session.outputNames[0]];
    detections    = postprocess(out.data, scale, padX, padY, origW, origH);
    for (const t of Object.values(results)) t.dispose?.();
  } catch (err) {
    console.error('inference error:', err);
  } finally {
    tensor.dispose?.();
  }
  // Always reply so app.js resets inferring=false, even on error
  self.postMessage({ type: 'result', detections });
};

loadModel();
