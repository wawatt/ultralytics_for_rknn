## build
```shell
.\env\python.exe -m pip uninstall ultralytics
.\env\python.exe -m pip install  -e .


.\env\python.exe workspace/train_yolov12_dino.py --data ./ultralytics/cfg/datasets/construction-ppe.yaml --yolo-size m --dinoversion 3 --dino-variant vitb16 --dino-input E:/Pro/github/anomalib/workspace/pre_trained/dinov3_vitb16_pretrain_lvd1689m.pth --integration dualp0p3 --epochs 100 --batch-size 8 --name m_vitb16_dualp0p3

.\env\python.exe workspace/train_yolov12_dino.py --data ./ultralytics/cfg/datasets/construction-ppe.yaml --yolo-size s --dinoversion 3 --dino-variant vitb16 --dino-input E:/Pro/github/anomalib/workspace/pre_trained/dinov3_vitb16_pretrain_lvd1689m.pth --integration dualp0p3 --epochs 100 --batch-size 8 --name s_vitb16_dualp0p3

.\env\python.exe workspace/train_yolov12_dino.py --data ./ultralytics/cfg/datasets/construction-ppe.yaml --yolo-size m --dinoversion 3 --dino-variant convnext_base --integration single --epochs 100 --batch-size 8 --name s_convnext_small_single

.\env\python.exe workspace/train_yolov12_dino.py --data ./ultralytics/cfg/datasets/construction-ppe.yaml --yolo-size m --epochs 50 --batch-size 8 --name m

```