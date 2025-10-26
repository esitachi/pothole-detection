# Pothole Detection - Project

This repository was prepared from the uploaded project zip.

## Structure
```
/mnt/data/pothole-detection-ready
├── app/                      # Streamlit app
│   └── app.py
├── models/                   # Training & test scripts (from your upload)
├── runs/                     # Trained model weights (kept here)
│   └── detect/train/weights/best.pt
├── dataset/                  # (copied if present in your upload)
├── requirements.txt
└── .gitignore
```

## Run the Streamlit app (locally)
1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app/app.py
   ```

## Notes
- The app expects the trained model at `runs/detect/train/weights/best.pt`. I copied any `.pt` file I found into that path.
- If your `best.pt` is large, it is included in the final package you downloaded here.
- For GitHub: large files (over 100 MB) should be handled via Git LFS or hosted externally.

