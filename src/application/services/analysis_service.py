import io
import uuid
import numpy as np
import cv2
from datetime import datetime
import tensorflow as tf
import json
from pathlib import Path
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from src.infrastructure.sql_db import insert_report_group
from src.infrastructure.mongo_db import save_analysis_group_safe


BASE = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE / "models"
MODEL_PATH = MODEL_DIR / "best_model.keras"
CLASS_PATH = MODEL_DIR / "class_indices.json"
IMG_SIZE = (128, 128)

# --- Cargar modelo CNN y etiquetas ---
print("[IA] Cargando modelo CNN...")
model = tf.keras.models.load_model(str(MODEL_PATH))
with open(CLASS_PATH, "r", encoding="utf-8") as f:
    idx2class = json.load(f)
print("[IA] Modelo cargado correctamente.")


# --- Preprocesamiento ---
def preprocess_image(file_bytes):
    """Convierte los bytes de imagen en un arreglo RGB redimensionado."""
    if not file_bytes:
        return None
    arr = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return img


# --- Predicción CNN ---
def analyze_image(img):
    """Ejecuta el modelo CNN para clasificar la imagen."""
    

    normalized_img = img.astype('float32') / 255.0 
    
    preds = model.predict(np.expand_dims(normalized_img, 0))[0] 
    
    idx = int(np.argmax(preds))
    label = idx2class.get(str(idx), f"Clase {idx}")
    return label, float(preds[idx]), preds.tolist()



def analyze_files_preview(files, generated_by, sample_id, observations=""):
    """
    Procesa múltiples imágenes, aplica IA + morfología, guarda en DB, y devuelve resumen con features.
    """
    print(f"[SERVICE] Iniciando análisis de {len(files)} imágenes...")

    results = []
    total_prob = 0.0
    final_label = None
    preds_all = []
    features_all = {} 

    images_data = []

    for uploaded_file in files:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.filename or "unknown.jpg"
        if not filename or filename.strip() == "":  
            filename = "unknown.jpg"
        print(f"[SERVICE] Filename procesado: {repr(filename)}")
        
        img = preprocess_image(file_bytes)
        if img is None:
            print(f"[WARN] No se pudo procesar la imagen {filename}")
            continue
        
        label, prob, preds = analyze_image(img)
        features, overlay = analyze_morphology_visual(img) 

        total_prob += prob
        final_label = label
        preds_all.append(preds)
        features_all[filename] = features  

        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        overlay_bytes = buffer.tobytes()

        results.append({
            "filename": filename,
            "overlay_image": overlay_bytes,
            "predicted_class": label,
            "probability": prob,
            "probability_vector": preds,
            "features": features
        })

        
        images_data.append((filename, overlay_bytes))
        overlay_filename = f"{filename.rsplit('.', 1)[0]}_overlay.jpg"
        images_data.append((overlay_filename, overlay_bytes)) 

    if not results:
        raise ValueError("No se pudieron procesar las imágenes correctamente.")

    avg_prob = total_prob / len(results)
    print(f"[SERVICE] Promedio de probabilidad: {avg_prob:.4f}")
    print(f"[SERVICE] Predicción final: {final_label}")

    # Guardar en Mongo y SQL (sin cambios)
    analysis_id, first_image_id = save_analysis_group_safe(
        files=images_data,
        predicted_class=final_label,
        probability=avg_prob,
        probability_vector=[float(p) for p in preds_all[0]],
        features=features_all
    )

    insert_report_group(
        sample_id=sample_id,
        generated_by=generated_by,
        predicted_class=final_label,
        probability=avg_prob,
        features=features_all,
        observations=f"Análisis grupal de {len(results)} imágenes. {observations}"
    )

    print(f"[SERVICE] Análisis completado y guardado correctamente. ID: {analysis_id}")

    return {
        "analysis_id": analysis_id,
        "predicted_class": final_label,
        "probability": avg_prob,
        "features": features_all,  
        "probability_vector": preds_all[0] if preds_all else [] 
    }
    

def analyze_morphology_visual(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {}, img

    main_contour = max(contours, key=cv2.contourArea)
    overlay = img.copy()

    # Contorno
    cv2.drawContours(overlay, [main_contour], -1, (0, 255, 0), 2)
    area = cv2.contourArea(main_contour)
    perimeter = cv2.arcLength(main_contour, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    size_ratio = area / (img.shape[0] * img.shape[1])

    # Manchas
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mean_color = np.mean(hsv[:, :, 0])
    std_color = np.std(hsv[:, :, 0])
    manchas = "Sí" if std_color > 20 else "No"
    if manchas == "Sí":
        mask_var = cv2.inRange(hsv, (0, 30, 0), (179, 255, 200))
        overlay[mask_var > 0] = (255, 0, 0)

    # Daños
    edges = cv2.Canny(gray, 100, 200)
    damage_ratio = np.sum(edges > 0) / area
    damage = "Rajaduras detectadas" if damage_ratio > 0.15 else "Sin daños visibles"
    if "Rajaduras" in damage:
        overlay[edges > 0] = (255, 0, 0)

    # Impurezas
    impurities = "No"
    for cnt in contours:
        if cnt is not main_contour and cv2.contourArea(cnt) > 50:
            impurities = "Sí"
            cv2.drawContours(overlay, [cnt], -1, (0, 0, 255), 2)

    metrics = {
        "Color medio (H)": round(mean_color, 2),
        "Variación de color": f"{std_color:.2f} (Manchas: {manchas})",
        "Tamaño relativo": f"{size_ratio * 100:.1f}%",
        "Circularidad": f"{circularity:.2f}",
        "Daños mecánicos": damage,
        "Impurezas": impurities,
    }

    return metrics, overlay


def generate_analysis_steps(label, prob, preds, img):
    """Genera gráficos para pasos 6-8 usando la predicción."""
    visuals = {}
    
    # Paso 6: Extracción de features
    embedding = preds
    plt.figure(figsize=(6, 4))
    plt.hist(embedding, bins=20, alpha=0.7, color='blue')
    plt.title("Distribución de Features (Probabilidades)")
    plt.xlabel("Valor de Feature")
    plt.ylabel("Frecuencia")
    buf_img = BytesIO()
    plt.savefig(buf_img, format='png', bbox_inches='tight')
    buf_img.seek(0)
    visuals['features_histogram'] = base64.b64encode(buf_img.read()).decode('utf-8')
    plt.close()
    
    # Paso 7: Predicción de probabilidades
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(preds)), preds, color='green')
    plt.title("Probabilidades por Clase")
    plt.xlabel("Clase")
    plt.ylabel("Probabilidad")
    plt.xticks(range(len(preds)), [idx2class.get(str(i), f"Clase {i}") for i in range(len(preds))], rotation=45)
    buf_img = BytesIO()
    plt.savefig(buf_img, format='png', bbox_inches='tight')
    buf_img.seek(0)
    visuals['probabilities_bar'] = base64.b64encode(buf_img.read()).decode('utf-8')
    plt.close()
    
    # Paso 8: Selección de la clase
    plt.figure(figsize=(6, 4))
    colors = ['red' if i == int(np.argmax(preds)) else 'green' for i in range(len(preds))]
    plt.bar(range(len(preds)), preds, color=colors)
    plt.title(f"Selección de Clase - Predicha: '{label}'")
    plt.xlabel("Clase")
    plt.ylabel("Probabilidad")
    plt.xticks(range(len(preds)), [idx2class.get(str(i), f"Clase {i}") for i in range(len(preds))], rotation=45)
    buf_img = BytesIO()
    plt.savefig(buf_img, format='png', bbox_inches='tight')
    buf_img.seek(0)
    visuals['selected_class_bar'] = base64.b64encode(buf_img.read()).decode('utf-8')
    plt.close()
    
    steps = {
        "step6": f"Extracción de features: Usa las probabilidades del modelo como features básicas ({len(embedding)} valores).",
        "step7": f"Predicción de probabilidades: Calcula probabilidades para cada clase usando el modelo entrenado.",
        "step8": f"Selección de la clase: Elige la clase con mayor probabilidad '{label}' con {prob*100:.2f}% de confianza."
    }
    
    return steps, visuals

def predict_step_by_step(buf):
    print("[DEBUG] predict_step_by_step called")
    try:
        img = preprocess_image(buf)
        if img is None:
            print("[ERROR] Imagen inválida")
            return {"error": "Imagen inválida"}
        
        label, prob, preds = analyze_image(img)
        
        # Generar pasos 6-8
        steps_6_8, visuals_6_8 = generate_analysis_steps(label, prob, preds, img)
        
        steps = {
            "step6": steps_6_8["step6"],
            "step7": steps_6_8["step7"],
            "step8": steps_6_8["step8"]
        }
        
        return {
            "steps": steps,
            "label": label,"probability": prob,
            "embedding": preds[:10],
            "all_probabilities": preds,
            "visuals": visuals_6_8
        }
    except Exception as e:
        print(f"[ERROR] En predict_step_by_step: {e}")
        import traceback
        traceback.print_exc()
        return {"error": "Error interno del servidor"}