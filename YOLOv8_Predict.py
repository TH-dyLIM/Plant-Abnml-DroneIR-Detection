from ultralytics import YOLO
import cv2

# Load a pretrained YOLOv8n model
model = YOLO('./runs/detect/yolov8s_custom10/weights/best.pt')

# Define path to directory containing images and videos for inference
source = '비정상상태_SG 1 수외 0%.mp4'

# Run inference on the source
results = model(source, imgsz=1280, save=True, hide_conf=True)  # generator of Results objects
