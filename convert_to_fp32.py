"""
Konvertiert yolov5n.onnx vollständig von float16 → float32.
Ausführen: python convert_to_fp32.py
Benötigt:  pip install onnx numpy
"""
import onnx
from onnx import numpy_helper, TensorProto
import numpy as np

MODEL = 'models/yolov5n.onnx'

model = onnx.load(MODEL)

# 1) Gewichte (Initialisierer) konvertieren
for init in model.graph.initializer:
    if init.data_type == TensorProto.FLOAT16:
        fp32 = numpy_helper.to_array(init).astype(np.float32)
        init.CopyFrom(numpy_helper.from_array(fp32, init.name))

# 2) Typ-Annotationen aller Tensoren aktualisieren
for container in (model.graph.input, model.graph.output, model.graph.value_info):
    for t in container:
        if t.type.tensor_type.elem_type == TensorProto.FLOAT16:
            t.type.tensor_type.elem_type = TensorProto.FLOAT

# 3) Cast-Nodes die auf float16 zeigen → float32 umbiegen
for node in model.graph.node:
    if node.op_type == 'Cast':
        for attr in node.attribute:
            if attr.name == 'to' and attr.i == TensorProto.FLOAT16:
                attr.i = TensorProto.FLOAT

onnx.save(model, MODEL)
print(f"Fertig — {MODEL} ist jetzt float32.")
