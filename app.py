from pathlib import Path
import os
import uuid

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
UPLOAD_FOLDER = ROOT / "uploads"
RESULT_FOLDER = ROOT / "static" / "results"
MODEL_PATH = ROOT / "models" / "yolov5s.onnx"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45

COCO_NAMES = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog",
    "horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
    "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork",
    "knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog",
    "pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor",
    "laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["RESULT_FOLDER"] = str(RESULT_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modell nicht gefunden: {model_path}. Bitte lade 'yolov5s.onnx' in das Verzeichnis 'models' herunter."
        )
    net = cv2.dnn.readNet(str(model_path))
    return net


def prepare_image(image, size=(640, 640)):
    return cv2.dnn.blobFromImage(
        image,
        scalefactor=1 / 255.0,
        size=size,
        mean=(0, 0, 0),
        swapRB=True,
        crop=False,
    )


def decode_outputs(outputs, image_shape, conf_threshold, input_size=(640, 640)):
    image_height, image_width = image_shape[:2]
    input_width, input_height = input_size
    scale_x = image_width / input_width
    scale_y = image_height / input_height

    boxes = []
    confidences = []
    class_ids = []

    output = outputs[0]
    if output.ndim == 3:
        output = output[0]

    for detection in output:
        scores = detection[5:]
        class_id = int(np.argmax(scores))
        class_score = float(scores[class_id])
        objectness = float(detection[4])
        confidence = class_score * objectness
        if confidence < conf_threshold:
            continue

        cx, cy, w, h = detection[0:4]
        x = int((cx - w / 2) * scale_x)
        y = int((cy - h / 2) * scale_y)
        width = int(w * scale_x)
        height = int(h * scale_y)

        if width <= 0 or height <= 0:
            continue

        boxes.append([x, y, width, height])
        confidences.append(confidence)
        class_ids.append(class_id)

    return boxes, confidences, class_ids


def draw_boxes(image, boxes, confidences, class_ids, indices):
    for i in indices:
        x, y, w, h = boxes[i]
        class_id = class_ids[i]
        conf = confidences[i]
        label = f"{COCO_NAMES[class_id]}: {conf:.2f}"
        color = (0, 255, 0)

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            image,
            label,
            (x, y - 10 if y - 10 > 10 else y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )


def postprocess(image, outputs, conf_threshold, nms_threshold):
    boxes, confidences, class_ids = decode_outputs(outputs, image.shape, conf_threshold)
    if not boxes:
        return image, []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if len(indices) > 0:
        indices = indices.flatten().tolist()
    else:
        indices = []

    draw_boxes(image, boxes, confidences, class_ids, indices)
    detections = []
    for i in indices:
        detections.append({
            "class_id": class_ids[i],
            "class_name": COCO_NAMES[class_ids[i]],
            "confidence": confidences[i],
            "box": boxes[i],
        })
    return image, detections


def process_image(net, source_path: str, output_path: str):
    image = cv2.imread(source_path)
    if image is None:
        raise FileNotFoundError(f"Eingabebild nicht gefunden: {source_path}")

    blob = prepare_image(image)
    net.setInput(blob)
    outputs = net.forward()
    result, detections = postprocess(image, outputs, CONF_THRESHOLD, NMS_THRESHOLD)

    cv2.imwrite(str(output_path), result)
    return detections


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    upload_path = UPLOAD_FOLDER / filename
    file.save(str(upload_path))

    result_name = f"{uuid.uuid4().hex}_{filename}"
    result_path = RESULT_FOLDER / result_name

    net = load_model(MODEL_PATH)
    detections = process_image(net, str(upload_path), result_path)

    return render_template(
        "result.html",
        result_image=url_for("static", filename=f"results/{result_name}"),
        filename=filename,
        detections=detections,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
