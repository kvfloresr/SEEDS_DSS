from flask import Flask, request, jsonify
from pathlib import Path
import tensorflow as tf
import numpy as np
import json
import cv2

BASE = Path(__file__).resolve().parents[3]
MODEL_DIR = BASE / "models"
MODEL_PATH = MODEL_DIR / "seed_cnn.h5"
CLASS_PATH = MODEL_DIR / "class_indices.json"
IMG_SIZE = (128,128)

app = Flask(__name__)
model = tf.keras.models.load_model(str(MODEL_PATH))
with open(CLASS_PATH, "r", encoding="utf-8") as f:
    idx2class = json.load(f)

feature_extractor = tf.keras.Model(
    inputs=model.input,
    outputs=model.layers[-3].output
)

def preprocess_image_buffer(buf):
    arr = np.frombuffer(buf, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return img

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error":"image file required"}), 400
    f = request.files['image']
    img = preprocess_image_buffer(f.read())
    if img is None:
        return jsonify({"error":"invalid image"}), 400

    preds = model.predict(np.expand_dims(img,0))[0]
    idx = int(np.argmax(preds))
    label = idx2class[str(idx)] if str(idx) in idx2class else idx2class[idx]
    embedding = feature_extractor.predict(np.expand_dims(img,0))[0].tolist()

    return jsonify({
        "label": label,
        "probability": float(preds[idx]),
        "embedding": embedding
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
