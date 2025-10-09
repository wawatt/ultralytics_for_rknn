from ultralytics import YOLOE
from ultralytics.utils.patches import torch_load

det_model = YOLOE("yoloe-11l.yaml")
state = torch_load("yoloe-11l-seg.pt")
det_model.load(state["model"])
det_model.save("yoloe-11l-seg-det.pt")

# ------------------------- 开始训练 -------------------------
from ultralytics.models.yolo.yoloe import YOLOESegVPTrainer

data = dict(
    train=dict(
        yolo_data=["Objects365.yaml"],
        grounding_data=[
            dict(
                img_path="flickr/full_images/",
                json_file="flickr/annotations/final_flickr_separateGT_train_segm.json",
            ),
            dict(
                img_path="mixed_grounding/gqa/images",
                json_file="mixed_grounding/annotations/final_mixed_train_no_coco_segm.json",
            ),
        ],
    ),
    val=dict(yolo_data=["lvis.yaml"]),
)

model = YOLOE("yoloe-11l-seg.pt")
# replace to yoloe-11l-seg-det.pt if converted to detection model
# model = YOLOE("yoloe-11l-seg-det.pt")

# freeze every layer except of the savpe module.
head_index = len(model.model.model) - 1
freeze = list(range(0, head_index))
for name, child in model.model.model[-1].named_children():
    if "savpe" not in name:
        freeze.append(f"{head_index}.{name}")

model.train(
    data=data,
    batch=128,
    epochs=2,
    close_mosaic=2,
    optimizer="AdamW",
    lr0=16e-3,
    warmup_bias_lr=0.0,
    weight_decay=0.025,
    momentum=0.9,
    workers=4,
    trainer=YOLOESegVPTrainer,  # use YOLOEVPTrainer if converted to detection model
    device="0,1,2,3,4,5,6,7",
    freeze=freeze,
)