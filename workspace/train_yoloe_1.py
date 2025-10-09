
from ultralytics.models.yolo.yoloe import YOLOESegTrainerFromScratch
from ultralytics import YOLOE
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

model = YOLOE("yoloe-11s-seg.yaml")
model.train(
    data=data,
    batch=128,
    epochs=30,
    close_mosaic=2,
    optimizer="AdamW",
    lr0=2e-3,
    warmup_bias_lr=0.0,
    weight_decay=0.025,
    momentum=0.9,
    workers=0,
    trainer=YOLOESegTrainerFromScratch,
    device="0"
)