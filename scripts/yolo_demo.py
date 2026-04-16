import argparse
import os
from pathlib import Path

import cv2
import numpy as np

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO-ONNX-Objekterkennung mit OpenCV (lokal)."
    )
    parser.add_argument(
        "--model",
        default="models/yolov5s.onnx",
        help="Pfad zum YOLO-ONNX-Modell."
    )
    parser.add_argument(
        "--source",
        default="demo/demo.jpg",
        help="Eingabebild oder -video. Für Webcam: 0 oder 1."
    )
    parser.add_argument(
        "--output",
        default="output/result.jpg",
        help="Zielpfad für das Ergebnisbild oder -video."
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Minimaler Konfidenzwert für Erkennungen."
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.45,
        help="NMS-Schwelle zum Filtern überlappender Boxen."
    )
    return parser.parse_args()


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modell nicht gefunden: {model_path}. Bitte lade 'yolov5s.onnx' in das Verzeichnis 'models' herunter."
        )
    net = cv2.dnn.readNet(model_path)
    return net


def prepare_image(image, size=(640, 640)):
    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1 / 255.0,
        size=size,
        mean=(0, 0, 0),
        swapRB=True,
        crop=False,
    )
    return blob


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


def process_image(net, source_path: str, output_path: str, args):
    image = cv2.imread(source_path)
    if image is None:
        raise FileNotFoundError(f"Eingabebild nicht gefunden: {source_path}")

    blob = prepare_image(image)
    net.setInput(blob)
    outputs = net.forward()
    result, detections = postprocess(image, outputs, args.conf_threshold, args.nms_threshold)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, result)
    print(f"Ergebnis gespeichert in: {output_path}")
    print(f"Gefundene Objekte: {len(detections)}")
    for det in detections:
        print(f"- {det['class_name']} ({det['confidence']:.2f})")


def process_video(net, source_path: str, output_path: str, args):
    capture = cv2.VideoCapture(source_path)
    if not capture.isOpened():
        raise RuntimeError(f"Videoquelle konnte nicht geöffnet werden: {source_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    while True:
        has_frame, frame = capture.read()
        if not has_frame:
            break

        blob = prepare_image(frame)
        net.setInput(blob)
        outputs = net.forward()
        result = postprocess(frame, outputs, args.conf_threshold, args.nms_threshold)
        writer.write(result)

    capture.release()
    writer.release()
    print(f"Ergebnisvideo gespeichert in: {output_path}")


def run():
    args = parse_args()
    net = load_model(args.model)

    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    if isinstance(source, int):
        print("Webcam-Stream wird nicht in GitHub Pages gezeigt. Bitte lokal ausführen.")
        process_video(net, source, args.output, args)
    elif Path(source).suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        process_image(net, source, args.output, args)
    else:
        process_video(net, source, args.output, args)


if __name__ == "__main__":
    run()
