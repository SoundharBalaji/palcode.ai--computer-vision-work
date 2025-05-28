from ultralytics import YOLO

# Load pretrained YOLOv8n model
model = YOLO('yolov8s.pt')

# Train the model
train_results = model.train(
    data=r"E:\palcode.ai\dataset\data.yaml",  # dataset config
    epochs=50,
    imgsz=640,        # train at higher resolution (can go 768 or 960 if CPU allows)
    batch=4,          # lower batch for CPU-based training
    device="cpu",     
    save_dir=r"E:\palcode.ai",
    workers=0,        # avoid multithreaded dataloading on Windows
    close_mosaic=10,  # disables mosaic after 10 epochs for better small object learning
    mosaic=1.0,       # keep mosaic enabled initially
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  # strong augmentation
    flipud=0.5, fliplr=0.5,
    degrees=10, translate=0.1, scale=0.3, shear=2
)

# Verify the model's class names
print(model.model.names)  # Should output: {0: 'door', 1: 'window'}
