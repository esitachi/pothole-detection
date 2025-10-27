from ultralytics import YOLO

# Load YOLOv8 pre-trained model
model = YOLO("yolov8n.pt") 

# Train the model
model.train(
    data=r"C:\Users\eshaa\OneDrive\Desktop\pothole project\pothole2.yaml",  
    epochs=50,             
    imgsz=640,            
    batch=16               
)
