from ultralytics import YOLO

model = YOLO("yolo11m.yaml")
model.train(
    data="./ultralytics/cfg/datasets/construction-ppe.yaml",
    batch=8,
    epochs=50,
    seed=0,
    optimizer="SGD",
    scale=0.9,
    mixup=0.15,
    workers=0,
    deterministic=False,
    overlap_mask=False,
    resume=None,
    name="yolo11m",
    device="0"
)