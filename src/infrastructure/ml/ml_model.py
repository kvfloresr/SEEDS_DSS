import tensorflow as tf
import numpy as np
import cv2
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "seed_cnn.h5"
CLASS_PATH = BASE / "models" / "class_indices.json"

model = tf.keras.models.load_model(str(MODEL_PATH))
with open(CLASS_PATH, "r", encoding="utf-8") as f:
    idx2class = json.load(f)

IMG_SIZE = (128, 128)

def preprocess_image(file_bytes):
    arr = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return img

def analyze_image(img):
    preds = model.predict(np.expand_dims(img, 0))[0]
    idx = int(np.argmax(preds))
    label = idx2class.get(str(idx), f"Clase {idx}")
    return label, float(preds[idx]), preds.tolist()
