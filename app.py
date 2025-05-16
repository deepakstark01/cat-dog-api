import os

import cv2
import numpy as np
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# Create FastAPI app
app = FastAPI()
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3000/",
    "http://localhost:4000",
    "http://localhost:4000/",
    "https://cat-dog-coral.vercel.app",
    "https://cat-dog-coral.vercel.app/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Determine the path to the model file relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "64x3-CNN.keras")

# Load your trained CNN model once at startup
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = tf.keras.models.load_model(model_path)

CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 50

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported file type")

    # Read image bytes and decode to grayscale
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # Preprocess image
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized / 255.0
    input_arr = img_norm.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # Perform prediction
    pred = model.predict(input_arr)
    label_idx = int(pred[0][0] >= 0.5)
    confidence = float(pred[0][0])

    return JSONResponse({"label": CATEGORIES[label_idx], "confidence": confidence})
@app.get("/")
async def root():
    return JSONResponse({"message": "Welcome to the Dog vs Cat API!"})
@app.get("/health")
async def health():
    return JSONResponse({"status": "healthy"})
@app.get("/model")
async def model_info():
    return JSONResponse({"model": "64x3-CNN", "version": "1.0"})

if __name__ == "__main__":
    import uvicorn
    # If this file is main.py, uvicorn.run can take the app instance directly:
    port = int(os.environ.get("PORT", 80))
    uvicorn.run(app, host="0.0.0.0", port=port)
