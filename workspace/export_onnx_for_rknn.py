from ultralytics import YOLO

# model = YOLO("yolo11s-seg.yaml").load("weights/yolo11s-seg.pt")
model = YOLO("yolo11s.yaml").load("weights/yolo11s-seg.pt")

# model.export(format="onnx", dynamic=False, simplify=True, opset=19, device="cpu")
# model.export(format="rkonnx", dynamic=False, simplify=True, opset=19, device="cpu")
model.export(format="rknn",  dynamic=False, simplify=True, opset=19, name="rk3588")
# 