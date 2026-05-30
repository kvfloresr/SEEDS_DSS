import uuid
import streamlit as st
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from src.infrastructure.mongo_db import save_analysis_group, get_reports
from src.infrastructure.sql_db import insert_report_group
from src.application.utils.feature_extraction import extract_morphological_features
from presentation.reports.pdf_report_generator import generate_pdf_report_full  

BASE = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE / "models"
MODEL_PATH = MODEL_DIR / "seed_cnn.h5"
CLASS_PATH = MODEL_DIR / "class_indices.json"
DATA_DIR = BASE / "data" / "processed"
IMG_SIZE = (128, 128)

# Cargar modelo
model = tf.keras.models.load_model(str(MODEL_PATH))
with open(CLASS_PATH, "r", encoding="utf-8") as f:
    idx2class = json.load(f)


# --- FUNCIONES ---
def preprocess_image(file_bytes):
    if not file_bytes:
        return None
    arr = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return img


def analyze_image(img):
    preds = model.predict(np.expand_dims(img, 0))[0]
    idx = int(np.argmax(preds))
    label = idx2class.get(str(idx), f"Clase {idx}")
    return label, preds[idx], preds


def compute_confusion_matrix():
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "test",
        image_size=IMG_SIZE,
        batch_size=32,
        label_mode="int",
        shuffle=False,
    )
    y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
    x_all = np.concatenate([x.numpy() for x, y in test_ds], axis=0)
    y_prob = model.predict(x_all, batch_size=32)
    y_pred = np.argmax(y_prob, axis=1)

    all_labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    labels = [idx2class.get(str(i), f"Clase {i}") for i in all_labels]
    return cm, labels

# Cargar modelo
model = tf.keras.models.load_model(str(MODEL_PATH))
with open(CLASS_PATH, "r", encoding="utf-8") as f:
    idx2class = json.load(f)


# --- FUNCIONES ---

def preprocess_image(file_bytes):
    if not file_bytes:
        return None
    arr = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return img


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





# --- INTERFAZ ---
st.set_page_config(page_title="Seed Quality DSS", layout="wide")
st.title("Sistema de Verificación Visual de Semillas de Soya con IA")

uploaded_files = st.file_uploader(
    "Sube una o varias imágenes de semillas para un solo análisis",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True
)

if uploaded_files:
    # --- Proceso de imágenes ---
    st.info("Todas las imágenes se analizarán juntas en un solo informe.")
    images_data = []
    features_all = {}
    preds_all = []
    mean_prob = 0
    label_final = ""

    # Analizar
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        img = preprocess_image(file_bytes)
        if img is None:
            st.error(f"No se pudo procesar la imagen {uploaded_file.name}")
            continue

        label, prob, preds = analyze_image(img)
        features, overlay = analyze_morphology_visual(img)

        mean_prob += prob
        preds_all.append(preds)
        features_all[uploaded_file.name] = features
        label_final = label  

        images_data.append((uploaded_file.name, file_bytes))

        tab1, tab2, tab3 = st.tabs([" Imagen & IA", " Morfología", "Matriz de Confusión"])

        # TAB 1
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption=f"Imagen: {uploaded_file.name}", width=550)
            with col2:
                st.subheader("Clasificación IA")
                st.success(f"**Clase predicha:** {label}")
                st.metric("Confianza del modelo", f"{prob* 100:.2f}%")
                st.bar_chart(preds)

        with tab2:
            st.subheader("Características morfológicas extraídas")
            col1, col2 = st.columns([1, 1.2]) 
            with col1:
                df = pd.DataFrame(list(features.items()), columns=["Parámetro", "Valor"])
                st.dataframe(df, use_container_width=True, height=250)
            with col2:
                st.image(overlay, caption="Análisis morfológico", width=520)
                st.markdown("""
                **Verde:** contorno de la semilla  
                **Rojo:** manchas y daños mecánicos  
                **Azul:** impurezas externas  
                """)

                
        with tab2:
            st.subheader("Morfología")
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _,mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            features = extract_morphological_features(img, mask)
            overlay = img.copy()
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                main_c = max(contours, key=cv2.contourArea)
                cv2.drawContours(overlay, [main_c], -1, (0, 255, 0), 2)
            col1, col2 = st.columns([1, 1])
            with col1: df = pd.DataFrame(list(features.items()), columns=["Parámetro", "Valor"])
            st.table(df)
            with col2:
                st.image(overlay, caption="Capa de Análisis Morfológico", width=500)

        # TAB 3
        with tab3:
            st.subheader("Matriz de Confusión del Modelo")
            cm, labels = compute_confusion_matrix()
            fig, ax = plt.subplots(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot(ax=ax, cmap="Greens", colorbar=False)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels)
            plt.tight_layout()
            st.pyplot(fig)

    # --- GUARDAR ---
    if st.button("Guardar Informe Grupal", key="guardar_grupal"):
        try:
            avg_prob = mean_prob / len(uploaded_files)

            analysis_id, image_ids = save_analysis_group(
                files=images_data,
                predicted_class=label_final,
                probability=avg_prob,
                probability_vector=[float(p) for p in preds_all[0]],
                features=features_all
            )


            insert_report_group(
                sample_id=str(uuid.uuid4()),
                generated_by="Sistema IA",
                predicted_class=label_final,
                probability=avg_prob,
                pureza_fisica="Buena",
                defect_percentage=5.0,
                observations=f"Análisis grupal de {len(uploaded_files)} imágenes."
            )

            pdf_path = generate_pdf_report_full(analysis_id)
            st.success(f"Informe grupal guardado correctamente (ID: {analysis_id})")

            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    "Descargar informe PDF",
                    data=pdf_file,
                    file_name=f"Seed_Report_{analysis_id}.pdf",
                    mime="application/pdf"
                )

        except Exception as e:
            st.error(f"Error al guardar informe: {e}")

    # --- HISTORIAL ---
    st.subheader("Historial de análisis guardados")
    rows = get_reports(limit=10)
    if rows:
        df_hist = pd.DataFrame.from_records(rows)
        st.dataframe(df_hist, use_container_width=True)
    else:
        st.info("Aún no hay informes guardados en la base de datos.")
