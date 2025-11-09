
import json
import numpy as np
import cv2
import os

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor, YOLOEVPDetectPredictor

def get_circle_points_range(center, radius, start_angle=0, end_angle=2*np.pi, num_points=100):
    """
    获取圆上指定角度范围内的点
    """
    cx, cy = center
    
    angles = np.linspace(start_angle, end_angle, num_points)
    
    x = cx + radius * np.cos(angles)
    y = cy + radius * np.sin(angles)
    
    points = np.column_stack((x, y))
    return points

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


    def add_refers_to_model(self, refer_images, visual_prompts):

        len_bboxes = len(visual_prompts["bboxes"])
        len_masks = len(visual_prompts["masks"])
        key_anno = ""
        len_anno = 0
        if len_bboxes > 0:
            len_anno = len_bboxes
            key_anno = "bboxes"
            print("-----------using bboxes")
        elif len_masks > 0:
            len_anno = len_masks
            key_anno = "masks"
            print("-----------using masks")
        else:
            raise ValueError("Please provide valid bboxes or masks")

        if len_anno != len(visual_prompts["cls"]):
            raise ValueError("The number of anno and cls must be equal")
        if len_anno != len(refer_images):
            raise ValueError("The number of anno and refer_images must be equal")

        if len(visual_prompts[key_anno]) == 1:
            print("Only one prompt, using the first prompt for first images")
            visual_prompts_tmp = dict(
                cls=visual_prompts["cls"][0]
            )
            visual_prompts_tmp[key_anno] = visual_prompts[key_anno][0]
            refer_images_tmp = refer_images[0]
        else:
            visual_prompts_tmp = dict(
                cls=visual_prompts["cls"],  
            )
            visual_prompts_tmp[key_anno] = visual_prompts[key_anno]
            refer_images_tmp = refer_images
    
        results = self.model.predict(
            source=refer_images[0],
            refer_image=refer_images_tmp,
            visual_prompts=visual_prompts_tmp,
            predictor=self.mode,
            imgsz=self.imgsz
        )
        results[0].show()

    def get_anno(self, labelme_json_path):
        """
        从LabelMe JSON文件中读取边界框坐标、标签类别和图像文件名
        
        Args:
            labelme_json_path (str): LabelMe JSON文件路径
            
        Returns:
            tuple: (bboxes, masks, labels, image_path) 
                bboxes: 边界框坐标，np.array格式，形状为(N, 4)，格式为[x_min, y_min, x_max, y_max]，数据类型为float32
                masks: 255
                labels: 标签类别列表，格式: ['label1', 'label2', ...]
                image_path: 图像文件路径
        """
        with open(labelme_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        labels_str = []
        bboxes = []
        masks = []

        # 图像文件路径：json文件中的imagePath字段，拼接json的文件夹路径获得完整路径
        json_dir = os.path.dirname(labelme_json_path)
        image_path = os.path.join(json_dir, data.get('imagePath', ''))
        img_height = data.get('imageHeight', 0)
        img_width = data.get('imageWidth', 0)

        for shape in data.get('shapes', []):
            label = shape.get('label')
            shape_type = shape.get('shape_type')
            points = shape.get('points')  # list of [x, y]
            if not points or len(points) == 0:
                print(f"Warning: shape {shape_type} with no points, skip")
                continue
            if shape_type not in ['polygon', 'rectangle']:
                print(f"Warning: shape {shape_type} not in ['polygon', 'rectangle'], skip")
                continue

            if self.mode == YOLOEVPDetectPredictor:
                # rectangle + YOLOEVPDetectPredictor or polygon + YOLOEVPDetectPredictor
                # 计算边界框，找到xmin, ymin, xmax, ymax
                xs = [point[0] for point in points]
                ys = [point[1] for point in points]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                print(f"bbox: {[x_min, y_min, x_max, y_max]}")
                bboxes.append([x_min, y_min, x_max, y_max])

            if shape_type == 'polygon' and self.mode == YOLOEVPSegPredictor:
                # 创建掩码
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
                masks.append(mask)

            labels_str.append(label)

        if len(labels_str)==0:
            return None, None, None, image_path
        
        label_mapping = {label: idx for idx, label in enumerate(sorted(set(labels_str)))}
        labels_int = [label_mapping[label] for label in labels_str]
        labels = np.array(labels_int, dtype=np.int32)

        if len(bboxes) != 0:
            bboxes = np.array(bboxes, dtype=np.float32)
            masks = None
        elif len(masks) != 0:
            bboxes = None
            masks = np.array(masks, dtype=np.uint8)
        else:
            raise ValueError(f"Please provide valid bboxes or masks")


        return bboxes, masks, labels, image_path
    
    def regist_refers(self, refer_dir="refers"):

        json_files = get_json_files(refer_dir)
        if len(json_files) == 0:
            raise ValueError(f"Refer dir {refer_dir} is empty")
        
        refer_images = []
        visual_prompts=dict(
            cls=[],
            bboxes=[],
            masks=[],
        )

        for json_file in json_files:
            bboxes, masks, labels, image_path = self.get_anno(json_file)
            print(bboxes)
            if labels is not None:
                print(image_path)
                
                refer_images.append(image_path)
                visual_prompts["cls"].append(labels)
            if bboxes is not None:
                visual_prompts["bboxes"].append(bboxes)
            if masks is not None:
                visual_prompts["masks"].append(masks)

        self.add_refers_to_model(refer_images, visual_prompts)


if __name__ == "__main__":
    model  = RegisterYOLOE(arch='s', imgsz=(640, 640), mode='det')
    # add refers
    model.regist_refers(refer_dir="refers/Pokemon")

    # 导出使用
    # model.export(format="rknn")
    model.export(format="onnx")
    # model.export(format="rkonnx")
    # model2 = YOLO("yoloe-11s-seg.onnx")
    # results=model2("assets/test3.jpg")
    # results[0].show(masks=False)
