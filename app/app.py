import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Pothole Detection", layout="wide")
st.title("🚗 Pothole Detection (YOLOv8)")

st.markdown("""
Upload a road image and the app will run pothole detection using your trained YOLOv8 model.
""")

MODEL_PATH = "runs/detect/train/weights/best.pt"

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model at '{MODEL_PATH}'. Make sure `best.pt` exists in the path inside the project.\\nError: {e}")
    model = None

uploaded_file = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])
if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)
    if st.button("Detect Potholes"):
        # run prediction
        results = model.predict(source=np.array(image), imgsz=640, conf=0.25)
        # results[0].plot() returns an array in many ultralytics versions
        try:
            annotated = results[0].plot()
            # annotated may be numpy array
            if isinstance(annotated, (list, tuple)):
                # sometimes plot returns list, take first
                annotated = annotated[0]
            st.image(annotated, caption="Detections", use_column_width=True)
        except Exception as e:
            st.error(f"Could not render results: {e}")
            st.write(results)
else:
    if not model:
        st.info("Place your trained model at: runs/detect/train/weights/best.pt and refresh.")
    else:
        st.info("Upload an image to get started.")
