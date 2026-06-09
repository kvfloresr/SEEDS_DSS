import os
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from sklearn.metrics import confusion_matrix
import cv2

from src.application.services.analysis_service import analyze_image as analyze_image_service
from src.application.services.analysis_service import preprocess_image


BASE = Path(__file__).resolve().parents[2]

DATA_DIR = BASE / "data" / "processed" / "test"
MODEL_PATH = BASE / "models" / "best_model.keras"  
CLASS_PATH = BASE /"src" / "models" / "class_indices.json"  

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

model = load_model(MODEL_PATH)
with open(CLASS_PATH, "r", encoding="utf-8") as f:
    idx2class = json.load(f)

feature_extractor = tf.keras.Model(
    inputs=model.input,
    outputs=model.layers[-3].output
)

def load_test_dataset():
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    print("\nClases detectadas:", test_ds.class_names)
    return test_ds, test_ds.class_names

class DataGeneratorIterator:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return sum(batch[0].shape[0] for batch in self.dataset)

    def __iter__(self):
        for batch in self.dataset:
            yield batch[0]



def preprocess_image_buffer(buf):
    arr = np.frombuffer(buf, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return img

# --- Predicción CNN ---
def analyze_image(img):
    """Ejecuta el modelo CNN para clasificar la imagen."""
    preds = model.predict(np.expand_dims(img, 0))[0]
    idx = int(np.argmax(preds))
    label = idx2class.get(str(idx), f"Clase {idx}")
    return label, float(preds[idx]), preds.tolist()

def evaluate_real_model(user="Sistema"):
    print("\n========== EVALUACIÓN REAL DEL MODELO ==========\n")
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Fecha de evaluación: {now}")
    print(f"Ejecutado por: {user}")
    
    # Cargar dataset
    test_ds, class_names = load_test_dataset()
    
    print("\nCargando modelo CNN...")
    model_loaded = load_model(MODEL_PATH)
    print("Modelo cargado.")
    
    # Evaluación interna
    print("\n===== MÉTRICAS INTERNAS (KERAS evaluate) =====")
    start_eval = time.time()
    loss, accuracy = model_loaded.evaluate(test_ds, verbose=1)
    keras_time = time.time() - start_eval
    
    print(f"Loss real:        {loss:.4f}")
    print(f"Accuracy real:    {accuracy:.4f}")
    print(f"Tiempo evaluate:  {keras_time:.4f} s")
    
    # Predicción
    print("\n===== GENERANDO PREDICCIONES =====")
    start_pred = time.time()
    y_pred_prob = model_loaded.predict(test_ds)
    pred_time = time.time() - start_pred
    
    # Tiempos por imagen
    num_images = sum(1 for _ in DataGeneratorIterator(test_ds))
    time_per_image = pred_time / num_images
    
    # Targets reales
    y_true = np.concatenate([y for _, y in test_ds], axis=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Métricas avanzadas
    print("\n===== MÉTRICAS AVANZADAS =====")
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    print(f"Precision macro:  {precision:.4f}")
    print(f"Recall macro:     {recall:.4f}")
    print(f"F1-score macro:   {f1:.4f}")
    print(f"Tiempo predicción:{pred_time:.4f} s")
    print(f"Tiempo por imagen:{time_per_image:.6f} s")
    
    # Reporte por clase
    print("\n===== CLASIFICATION REPORT =====")
    report = classification_report(
    y_true, y_pred,
    labels=list(range(len(class_names))), 
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)
    
    return {
        "timestamp": now,
        "user": user,
        "model_version": MODEL_PATH.name,
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "prediction_time_total": pred_time,
        "prediction_time_per_image": time_per_image,
        "evaluation_time": keras_time,
        "num_images": num_images,
        "class_names": class_names,
        "classification_report": report
    }

def predict_process(buf):
    print("[DEBUG] predict_step_by_step called")
    try:
        if model is None or feature_extractor is None:
            print("[ERROR] Modelo no cargado")
            return {"error": "Modelo no cargado"}
        
        img = preprocess_image_buffer(buf)
        if img is None:
            print("[ERROR] Imagen inválida")
            return {"error": "Imagen inválida"}
        
        visuals = {}
        
        # Paso 1: Conversión a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        plt.figure(figsize=(4, 4))
        plt.imshow(gray, cmap='gray')
        plt.title("Escala de Grises")
        plt.axis('off')
        buf_img = BytesIO()
        plt.savefig(buf_img, format='png', bbox_inches='tight')
        buf_img.seek(0)
        visuals['gray_image'] = base64.b64encode(buf_img.read()).decode('utf-8')
        plt.close()
        
        # Paso 2: Detección de bordes (Canny)
        edges = cv2.Canny(gray, 100, 200)
        plt.figure(figsize=(4, 4))
        plt.imshow(edges, cmap='gray')
        plt.title("Detección de Bordes (Canny)")
        plt.axis('off')
        buf_img = BytesIO()
        plt.savefig(buf_img, format='png', bbox_inches='tight')
        buf_img.seek(0)
        visuals['edges_image'] = base64.b64encode(buf_img.read()).decode('utf-8')
        plt.close()
        
        # Paso 3: Umbralización para detectar manchas
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        plt.figure(figsize=(4, 4))
        plt.imshow(thresh, cmap='gray')
        plt.title("Umbralización (Manchas)")
        plt.axis('off')
        buf_img = BytesIO()
        plt.savefig(buf_img, format='png', bbox_inches='tight')
        buf_img.seek(0)
        visuals['thresh_image'] = base64.b64encode(buf_img.read()).decode('utf-8')
        plt.close()
        
        # Paso 4: Segmentación (operaciones morfológicas)
        kernel = np.ones((5,5), np.uint8)
        segmented = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        plt.figure(figsize=(4, 4))
        plt.imshow(segmented, cmap='gray')
        plt.title("Segmentación (Morfología)")
        plt.axis('off')
        buf_img = BytesIO()
        plt.savefig(buf_img, format='png', bbox_inches='tight')
        buf_img.seek(0)
        visuals['segmented_image'] = base64.b64encode(buf_img.read()).decode('utf-8')
        plt.close()
        
        # Paso 5: Preprocesamiento final
        preprocessed_img = img / 255.0  # Normalización
        plt.figure(figsize=(4, 4))
        plt.imshow(preprocessed_img)
        plt.title("Preprocesamiento Final")
        plt.axis('off')
        buf_img = BytesIO()
        plt.savefig(buf_img, format='png', bbox_inches='tight')
        buf_img.seek(0)
        visuals['preprocessed_image'] = base64.b64encode(buf_img.read()).decode('utf-8')
        plt.close()
        
        
        # Pasos explicativos detallados
        steps = {
            "step1": "Conversión a escala de grises: Elimina información de color para enfocarse en intensidad, facilitando detección de patrones.",
            "step2": "Detección de bordes (Canny): Identifica contornos y bordes usando gradientes, crucial para definir formas de semillas.",
            "step3": "Umbralización: Aplica un umbral para binarizar la imagen, destacando manchas o áreas dañadas (e.g., spots en semillas).",
            "step4": "Segmentación: Usa operaciones morfológicas (cierre) para conectar componentes y segmentar objetos, separando semillas individuales.",
            "step5": "Preprocesamiento final: Redimensiona a 128x128 y normaliza (divide por 255), preparando para la CNN.",
            
        }
        
        return {
            "steps": steps,
            "visuals": visuals
        }
    except Exception as e:
        print(f"[ERROR] En predict_step_by_step: {e}")
        import traceback
        traceback.print_exc()
        return {"error": "Error interno del servidor"}

def evaluate_model(user="Sistema"):
    # 1) Métricas principales — si esto falla, sí es error real
    try:
        metrics = evaluate_real_model(user=user)
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": f"Error evaluando el modelo: {e}"}

    # 2) Métricas por clase (defensivo: si falta una clave, no rompe)
    class_metrics = []
    report = metrics.get("classification_report", {}) or {}
    for cls in metrics.get("class_names", []):
        r = report.get(cls)
        if isinstance(r, dict):
            class_metrics.append({
                "class": cls,
                "precision": r.get("precision", 0.0),
                "recall": r.get("recall", 0.0),
                "f1_score": r.get("f1-score", 0.0),
            })

    # 3) Matriz de confusión EN SU PROPIO try: si falla, igual devolvemos métricas
    cm_base64 = None
    try:
        ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR, image_size=IMG_SIZE, batch_size=32, shuffle=False
        )
        y_true = np.concatenate([y.numpy() for _, y in ds], axis=0)
        y_pred = np.argmax(model.predict(ds, verbose=0), axis=1)
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d',
            xticklabels=metrics["class_names"],
            yticklabels=metrics["class_names"],
            cmap='Blues'
        )
        plt.xlabel('Predicho')
        plt.ylabel('Real')
        plt.title('Matriz de Confusión')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        cm_base64 = "data:image/png;base64," + base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
    except Exception as e:
        print(f"[WARN] No se pudo generar la matriz de confusión: {e}")

    return {
        "metrics": metrics,
        "class_metrics": class_metrics,
        "confusion_matrix_base64": cm_base64,
    }

if __name__ == "__main__":
    metrics = evaluate_real_model(user="Valeria")
    print("\n====== EVALUACIÓN FINAL ======")
    print(json.dumps(metrics, indent=4, ensure_ascii=False))
