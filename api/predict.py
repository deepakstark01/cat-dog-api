import io
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

# Load your trained model once at cold start
model = tf.keras.models.load_model("64x3-CNN.keras")
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 50

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ensure we got an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported file type")

    # read bytes & decode into grayscale image
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # preprocess exactly as in training
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized / 255.0
    input_arr = img_norm.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # predict
    pred = model.predict(input_arr)
    label_idx = int(pred[0][0] >= 0.5)
    confidence = float(pred[0][0])

    return JSONResponse({
        "label": CATEGORIES[label_idx],
        "confidence": confidence
    })
