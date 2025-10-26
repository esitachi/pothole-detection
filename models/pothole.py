from ultralytics import YOLO

# Load YOLOv8 pre-trained model
model = YOLO("yolov8n.pt")  # you can also try yolov8s.pt for better accuracy

# Train the model
model.train(
    data=r"C:\Users\eshaa\OneDrive\Desktop\pothole project\pothole2.yaml",   # Dataset config
    epochs=50,             # Training epochs
    imgsz=640,             # Image size
    batch=16               # Adjust based on GPU
)