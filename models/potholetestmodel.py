from ultralytics import YOLO


model = YOLO(r"C:\Users\eshaa\OneDrive\Desktop\pothole project\runs\detect\train\weights")

# Test on a single image
results = model.predict(r"Pothole.v1-raw.yolov8/train/images/img-647_jpg.rf.47741041990f3eb84cbc316d5d91642d.jpg", show=True)

# test on multiple images
results = model.predict(source=r"C:\Users\eshaa\OneDrive\Desktop\pothole project\potholedataset\test\images", show=True)    
