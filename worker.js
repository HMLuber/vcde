/* YOLOv8n inference worker — runs in a separate thread so the UI never blocks */
'use strict';

importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.min.js');

const CONF_THRESH = 0.30;
const IOU_THRESH  = 0.45;
const ORT_CDN     = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/';

// INPUT_SIZE is set dynamically in warmup() to match whatever the model was
// exported with — no manual sync needed when swapping models.
let INPUT_SIZE = 160;

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

    // WebGPU: full GPU path, supports all YOLOv8 ops (incl. resize-nearest).
    // WebGL is intentionally skipped — it lacks resize-nearest support in ORT.
    // Falls back to WASM if WebGPU is unavailable (Firefox, older browsers).
    let provider = 'WASM';
    let gpuError = null;

    if (typeof navigator !== 'undefined' && navigator.gpu) {
      try {
        // Request the high-performance adapter — on dual-GPU laptops this selects
        // the discrete GPU (e.g. NVIDIA) instead of the default integrated GPU.
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) throw new Error('No WebGPU adapter found');

        // Pre-create the device and hand it to ORT so it runs on the same adapter.
        const gpuDevice = await adapter.requestDevice();
        ort.env.webgpu       = ort.env.webgpu ?? {};
        ort.env.webgpu.device = gpuDevice;

        session = await ort.InferenceSession.create('models/yolov8n.onnx', {
          executionProviders:     ['webgpu'],
          graphOptimizationLevel: 'all',
        });

        // Show which GPU was actually selected (e.g. "WebGPU · NVIDIA GeForce GTX 1650 Ti")
        const info = adapter.info ?? null;
        provider = info?.description ? `WebGPU · ${info.description}` : 'WebGPU';
      } catch (e) {
        gpuError = String(e);
      }
    } else {
      gpuError = 'navigator.gpu not available';
    }

    if (!session) {
      session = await ort.InferenceSession.create('models/yolov8n.onnx', {
        executionProviders:     ['wasm'],
        graphOptimizationLevel: 'all',
      });
    }

    // Warmup: auto-detects the model's actual input size, then runs 3 dummy
    // inferences to pre-compile all WebGPU shaders before the first real frame.
    INPUT_SIZE = await warmup(session);

    self.postMessage({ type: 'ready', provider, gpuError, inputSize: INPUT_SIZE });
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err) });
  }
}

/* Runs dummy inferences to pre-compile GPU shaders.
   On the first attempt uses the current INPUT_SIZE; if the model rejects it
   (wrong export resolution), parses the expected size from the ORT error and
   retries once — so swapping models never requires a manual code edit.       */
async function warmup(sess) {
  let sz = INPUT_SIZE;
  for (let attempt = 0; attempt <= 1; attempt++) {
    const dummy = new ort.Tensor('float32', new Float32Array(3 * sz * sz), [1, 3, sz, sz]);
    try {
      for (let i = 0; i < 3; i++) {
        const r = await sess.run({ [sess.inputNames[0]]: dummy });
        for (const t of Object.values(r)) t.dispose?.();
      }
      dummy.dispose?.();
      return sz; // success — this is the correct size
    } catch (e) {
      dummy.dispose?.();
      if (attempt === 0) {
        // Parse "Expected: 320" from the ORT dimension-mismatch error
        const m = String(e).match(/Expected:\s*(\d+)/);
        if (m) {
          sz = parseInt(m[1]);
          console.warn(`[worker] INPUT_SIZE auto-corrected: ${INPUT_SIZE} → ${sz}`);
          continue;
        }
      }
      throw e; // unrecoverable
    }
  }
  return sz;
}

/* ---- Post-processing: decode YOLOv8 output tensor ----------
   Handles both layouts that ultralytics may export:
     [1, 84, N]  — rows = attributes, cols = anchors
     [1, N, 84]  — rows = anchors,    cols = attributes (transposed)
   N_DET is read from the actual tensor dims, not hardcoded.       */
function postprocess(out, scale, padX, padY, origW, origH) {
  const [, d1, d2] = out.dims;
  const raw = out.data;

  // If d1 === 84 → standard layout [1, 84, N]; otherwise transposed [1, N, 84]
  const transposed = (d1 !== 84);
  const nDet       = transposed ? d1 : d2;

  const get = transposed
    ? (row, col) => raw[col * 84   + row]   // [N, 84]: anchor-major
    : (row, col) => raw[row * nDet + col];  // [84, N]: attribute-major

  const boxes = [], scores = [], classIds = [];

  for (let i = 0; i < nDet; i++) {
    let maxScore = 0, classId = 0;
    for (let c = 0; c < 80; c++) {
      const s = get(4 + c, i);
      if (s > maxScore) { maxScore = s; classId = c; }
    }
    if (maxScore < CONF_THRESH) continue;

    const cx = get(0, i), cy = get(1, i);
    const bw = get(2, i), bh = get(3, i);

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
    detections    = postprocess(out, scale, padX, padY, origW, origH);
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
