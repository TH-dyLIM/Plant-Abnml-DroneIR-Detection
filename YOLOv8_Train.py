import torch

# GPU 사용 가능 -> True, GPU 사용 불가 -> False
print(torch.cuda.is_available())

from ultralytics import YOLO

# Load the model.
model = YOLO('yolov8s.pt')

# Training.
def main():
    results = model.train(
        data='customdata.yaml',
        imgsz=640,
        epochs=400,
        batch=16,
        name='yolov8s_custom13',
        task='detect',
        mode='train',
        save_dir='runs\detect\yolov8s_custom13',
        patience=50,
        save=True,
        save_period=-1,
        device=0,
        workers=8,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=0,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        amp=True,
        fraction=1.0,
        profile=False,
        overlap_mask=True,
        val=True,
        split='val',
        save_json=False,
        save_hybrid=False,
        conf=None, iou=0.7, max_det=300,
        half=False, dnn=False, plots=True,
        source=None, show=False, save_txt=False,
        save_conf=False, save_crop=False, show_labels=True,
        show_conf=True, vid_stride=1, stream_buffer=False,
        line_width=None, visualize=False, augment=False,
        agnostic_nms=False, classes=None, retina_masks=False, boxes=True,
        format='torchscript', keras=False, optimize=False, int8=False, dynamic=False,
        simplify=False, opset=None, workspace=4, nms=False,
        lr0=0.001, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
        warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0,
        label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1,
        scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0,
        copy_paste=0.0, cfg=None, tracker='botsort.yaml'
    )
if __name__ == '__main__':

    main()