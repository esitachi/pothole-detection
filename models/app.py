import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("🚗 Pothole Detection System")

model = YOLO("runs/detect/train/weights/best.pt")

uploaded_file = st.file_uploader("Upload a road image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Potholes"):
        results = model.predict(img)
        result_img = results[0].plot()
        st.image(result_img, caption="Detected Potholes", use_column_width=True)
