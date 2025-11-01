
import json
import numpy as np
import cv2
import os

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor, YOLOEVPDetectPredictor

def get_anno(labelme_json_path):
    """
    从LabelMe JSON文件中读取边界框坐标、标签类别和图像文件名
    
    Args:
        labelme_json_path (str): LabelMe JSON文件路径
        
    Returns:
        tuple: (bboxes, labels, image_path) 
               bboxes: 边界框坐标，np.array格式，形状为(N, 4)，格式为[x_min, y_min, x_max, y_max]，数据类型为float32
               labels: 标签类别列表，格式: ['label1', 'label2', ...]
               image_path: 图像文件路径
    """
    with open(labelme_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    labels_str = []
    bboxes = []

    # 图像文件路径：json文件中的imagePath字段，拼接json的文件夹路径获得完整路径
    json_dir = os.path.dirname(labelme_json_path)
    image_path = os.path.join(json_dir, data.get('imagePath', ''))
    
    for shape in data.get('shapes', []):
        label = shape.get('label')
        points = shape.get('points')  # list of [x, y]
        if not points or len(points) == 0:
            continue
        
        # 计算边界框，找到xmin, ymin, xmax, ymax
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        bboxes.append([x_min, y_min, x_max, y_max])
        labels_str.append(label)
    if len(labels_str)==0:
        return None, None, image_path
    
    label_mapping = {label: idx for idx, label in enumerate(sorted(set(labels_str)))}
    labels_int = [label_mapping[label] for label in labels_str]
    labels = np.array(labels_int, dtype=np.int32)

    bboxes = np.array(bboxes, dtype=np.float32)


    return bboxes, labels, image_path

def get_json_files(dir):
    import os
    json_files = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files

class RegisterYOLOE:
    def __init__(self, arch: str = "s", imgsz: tuple=(640, 640), mode='det'): # s m l
        self.model = None
        if mode == 'det':
            self.mode = YOLOEVPDetectPredictor
            # https://github.com/ultralytics/ultralytics/issues/21581
            self.model = YOLOE(f"yoloe-11{arch}.yaml").load(f"weights/yoloe-11{arch}-seg.pt")
        elif mode == 'seg':
            self.mode = YOLOEVPSegPredictor
            self.model = YOLOE(f"yoloe-11{arch}-seg.yaml").load(f"weights/yoloe-11{arch}-seg.pt")
            
        self.model.overrides['imgsz'] = imgsz
        self.imgsz = imgsz
        self.arch = arch

    def __call__(self, source):
        return self.model(source)

    def export(self, format="rkonnx", taget_platform="rk3588"):
        opset_version = 19
        if format=="onnx":
            self.model.export(format="onnx", imgsz=self.imgsz, dynamic=True, simplify=True, opset=opset_version, device="cpu")
        elif format=="rkonnx":
            self.model.export(format="rkonnx", imgsz=self.imgsz, dynamic=True, simplify=True, opset=opset_version, device="cpu")
        elif format=="rknn":
            self.model.export(format="rknn", imgsz=self.imgsz, dynamic=False, simplify=True, opset=opset_version, name=taget_platform)
        else:
            raise ValueError(f"Unsupported format: {format}")


    def add_refers(self, refer_images, visual_prompts):
        if len(visual_prompts["bboxes"]) != len(visual_prompts["cls"]):
            raise ValueError("The number of bboxes and cls must be equal")
        if len(visual_prompts["bboxes"]) != len(refer_images):
            raise ValueError("The number of refer_images and visual_prompts must be equal")

        if len(visual_prompts["bboxes"]) == 1:
            print("Only one prompt, using the first prompt for first images")
            visual_prompts_tmp = dict(
                bboxes=visual_prompts["bboxes"][0],
                cls=visual_prompts["cls"][0],  
            )
            refer_images_tmp = refer_images[0]
        else:
            visual_prompts_tmp = dict(
                bboxes=visual_prompts["bboxes"],
                cls=visual_prompts["cls"],  
            )
            refer_images_tmp = refer_images
    
        results = self.model.predict(
            refer_images[0],
            refer_image=refer_images_tmp,
            visual_prompts=visual_prompts_tmp,
            predictor=self.mode,
            imgsz=self.imgsz
        )


if __name__ == "__main__":
    model  = RegisterYOLOE(arch='s', imgsz=(640, 640), mode='det')
    # add refers
    refer_images = []
    visual_prompts=dict(
        bboxes=[],
        cls=[],
    )
    json_files = get_json_files("refers")
    for json_file in json_files:
        print(json_file)
        bboxes, labels, image_path = get_anno(json_file)
        if bboxes is not None and labels is not None:
            refer_images.append(cv2.imread(image_path))
            visual_prompts["bboxes"].append(bboxes)
            visual_prompts["cls"].append(labels)
   
    model.add_refers(refer_images, visual_prompts)

    # json_files = get_json_files("assets/scew/h/test")
    # for json_file in json_files:
    #     bboxes, labels, image_path = get_anno(json_file)
    #     # refer_images.append(cv2.imread(image_path))
    #     # visual_prompts["bboxes"].append(bboxes)
    #     # visual_prompts["cls"].append(labels)

    #     results = model(image_path)
    #     results[0].show()

    #     break

    # 导出使用
    # model.export(format="rknn")
    model.export(format="onnx")
    # model.export(format="rkonnx")
    # model2 = YOLO("yoloe-11s-seg.onnx")
    # results=model2("assets/test3.jpg")
    # results[0].show(masks=False)
