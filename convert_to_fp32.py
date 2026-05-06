"""
Konvertiert yolov5n.onnx vollständig von float16 → float32.
Ausführen: python convert_to_fp32.py
Benötigt:  pip install onnx numpy
"""
import onnx
from onnx import numpy_helper, TensorProto, shape_inference
import numpy as np

MODEL = 'models/yolov5n.onnx'
model = onnx.load(MODEL)

# Shape Inference füllt value_info für ALLE Zwischen-Tensoren
model = shape_inference.infer_shapes(model)

# 1) Initialisierer (Gewichte) konvertieren
for init in model.graph.initializer:
    if init.data_type == TensorProto.FLOAT16:
        fp32 = numpy_helper.to_array(init).astype(np.float32)
        init.CopyFrom(numpy_helper.from_array(fp32, init.name))

# 2) Alle Tensor-Typ-Annotationen auf float32 setzen
for container in (model.graph.input, model.graph.output, model.graph.value_info):
    for t in container:
        if t.type.tensor_type.elem_type == TensorProto.FLOAT16:
            t.type.tensor_type.elem_type = TensorProto.FLOAT

# 3) Cast-Nodes die auf float16 zeigen → float32
for node in model.graph.node:
    if node.op_type == 'Cast':
        for attr in node.attribute:
            if attr.name == 'to' and attr.i == TensorProto.FLOAT16:
                attr.i = TensorProto.FLOAT

# 4) Constant-Nodes mit float16-Werten konvertieren
for node in model.graph.node:
    if node.op_type == 'Constant':
        for attr in node.attribute:
            if attr.name == 'value' and attr.t.data_type == TensorProto.FLOAT16:
                fp32 = numpy_helper.to_array(attr.t).astype(np.float32)
                attr.t.CopyFrom(numpy_helper.from_array(fp32))

onnx.save(model, MODEL)
print(f"Fertig — {MODEL} ist jetzt float32.")
