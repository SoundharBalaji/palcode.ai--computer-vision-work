import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np

# Load your trained model
model = YOLO("best.pt")  # Replace with your actual model path

def detect_objects(image):
    results = model.predict(image, imgsz=640)[0]
    detections = []
    for box in results.boxes:
        label = model.names[int(box.cls)]
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bbox = [x1, y1, x2 - x1, y2 - y1]
        detections.append({"label": label, "confidence": round(conf, 2), "bbox": bbox})
    return {"detections": detections}

gr.Interface(fn=detect_objects,
             inputs=gr.Image(type="numpy"),
             outputs="json",
             title="Door and Window Detection",
             description="Upload blueprint image to detect doors and windows").launch()
