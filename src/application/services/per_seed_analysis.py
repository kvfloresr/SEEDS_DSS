"""
per_seed_analysis.py — Análisis por semilla con criterios INIAF 2022.

Indicadores de calidad física (Tabla 3.1 INIAF):
  Pureza física     ≥ 98 %   Semillas intactas / total
  Materia inerte    ≤  2 %   Objetos no semilla (visión clásica)
  Daños mecánicos   ≤  1 %   Quebradas + Cubierta dañada (CNN)
  Color             Uniforme  Manchadas = 0 %
  Forma y tamaño    En rango  Inmaduras ≤ 2 %

Fuente: Compendio de Normas Nacionales sobre Semillas, INIAF 2022.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from src.live_camera import (
    segment, _circularity,
    DEFAULTS, CIRC_MIN, SIZE_LOW, SIZE_HIGH, COLOR_SAT_GATE,
)

BASE       = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE / "src" / "models" / "best_model.keras"
CLASS_PATH = BASE / "src" / "models" / "class_indices.json"
IMG_SIZE   = (128, 128)

# --- Normas INIAF 2022 -------------------------------------------------------
INIAF_PUREZA_MIN     = 98.0   # % mínimo de semillas intactas
INIAF_INERTE_MAX     = 2.0    # % máximo de materia inerte (no-soya)
INIAF_DANOS_MAX      = 1.0    # % máximo daños mecánicos (quebradas + cubierta dañada)
INIAF_MANCHADAS_MAX  = 0.0    # % para homogeneidad uniforme (ninguna manchada)
INIAF_INMADURAS_MAX  = 2.0    # % para forma y tamaño dentro del rango

# --- Colores y etiquetas -----------------------------------------------------
CLASS_COLORS = {
    "Intact soybeans":       "#22c55e",
    "Broken soybeans":       "#ef4444",
    "Immature soybeans":     "#f97316",
    "Skin-damaged soybeans": "#eab308",
    "Spotted soybeans":      "#8b5cf6",
}
CLASS_LABELS_ES = {
    "Intact soybeans":       "Intactas",
    "Broken soybeans":       "Quebradas / Rotas",
    "Immature soybeans":     "Inmaduras",
    "Skin-damaged soybeans": "Cubierta danada",
    "Spotted soybeans":      "Manchadas",
}

_model = None
_idx2class = None


def _get_model():
    global _model, _idx2class
    if _model is None:
        import tensorflow as tf
        _model = tf.keras.models.load_model(str(MODEL_PATH))
        with open(CLASS_PATH, "r", encoding="utf-8") as f:
            _idx2class = json.load(f)
        print("[CNN] Modelo cargado para analisis por semilla (INIAF).")
    return _model, _idx2class


def _crop_seed(frame_bgr, contour, x, y, w, h, margin=0.15):
    """Recorte simple (sin enmascarar fondo) para que coincida con lo que
    el modelo vio en entrenamiento: imagen de la semilla tal cual."""
    H, W = frame_bgr.shape[:2]
    cx, cy = x + w // 2, y + h // 2
    side   = int(max(w, h) * (1 + margin))
    half   = side // 2
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(W, cx + half), min(H, cy + half)
    crop = frame_bgr[y0:y1, x0:x1]
    return cv2.resize(crop, IMG_SIZE) if crop.size > 0 else None


def _detect_and_classify(img_bgr, model, idx2class):
    h, w = img_bgr.shape[:2]
    ratio = (w * h) / (640.0 * 480.0)
    p = {
        "min_area":  int(DEFAULTS["min_area"] * ratio),
        "max_area":  int(DEFAULTS["max_area"] * ratio),
        "thresh":    0, "invert": False, "color_sat": COLOR_SAT_GATE,
    }
    mask_img = segment(img_bgr, p["thresh"], p["invert"])
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = [(c, cv2.contourArea(c)) for c in contours
             if p["min_area"] <= cv2.contourArea(c) <= p["max_area"]]
    if not cands: return []
    med = sorted(a for _, a in cands)[len(cands) // 2]
    results = []
    for cnt, area in cands:
        if _circularity(cnt, area) < CIRC_MIN: continue
        if med > 0 and (area < SIZE_LOW * med or area > SIZE_HIGH * med): continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        crop = _crop_seed(img_bgr, cnt, x, y, bw, bh)
        if crop is None: continue
        rgb   = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype("float32")
        preds = model.predict(np.expand_dims(rgb, 0), verbose=0)[0]
        idx   = int(np.argmax(preds))
        results.append({
            "class":      idx2class.get(str(idx), f"Clase {idx}"),
            "confidence": round(float(preds[idx]), 3),
        })
    return results


def _build_iniaf_indicators(pcts_cnn, pct_inerte):
    """Construye la tabla de indicadores INIAF con valor, umbral y estado."""
    intact   = pcts_cnn.get("Intact soybeans",       0.0)
    broken   = pcts_cnn.get("Broken soybeans",        0.0)
    skindam  = pcts_cnn.get("Skin-damaged soybeans",  0.0)
    immature = pcts_cnn.get("Immature soybeans",      0.0)
    spotted  = pcts_cnn.get("Spotted soybeans",       0.0)
    danos    = broken + skindam

    color_ok = spotted <= INIAF_MANCHADAS_MAX
    shape_ok = immature <= INIAF_INMADURAS_MAX

    indicators = [
        {
            "name":      "Pureza fisica (%)",
            "value":     round(intact,   1),
            "threshold": f">= {INIAF_PUREZA_MIN:.0f} %",
            "norm":      INIAF_PUREZA_MIN,
            "pass":      intact >= INIAF_PUREZA_MIN,
            "icon":      "✓" if intact >= INIAF_PUREZA_MIN else "✗",
        },
        {
            "name":      "Materia inerte (%)",
            "value":     round(pct_inerte, 1),
            "threshold": f"<= {INIAF_INERTE_MAX:.0f} %",
            "norm":      INIAF_INERTE_MAX,
            "pass":      pct_inerte <= INIAF_INERTE_MAX,
            "icon":      "✓" if pct_inerte <= INIAF_INERTE_MAX else "✗",
        },
        {
            "name":      "Danos mecanicos (%)",
            "value":     round(danos,     1),
            "threshold": f"<= {INIAF_DANOS_MAX:.0f} %",
            "norm":      INIAF_DANOS_MAX,
            "pass":      danos <= INIAF_DANOS_MAX,
            "icon":      "✓" if danos <= INIAF_DANOS_MAX else "✗",
        },
        {
            "name":      "Homogeneidad de color",
            "value":     "Uniforme" if color_ok else "No uniforme",
            "threshold": "Uniforme",
            "norm":      None,
            "pass":      color_ok,
            "icon":      "✓" if color_ok else "✗",
        },
        {
            "name":      "Forma y tamano",
            "value":     "Dentro del rango" if shape_ok else "Fuera del rango",
            "threshold": "Dentro del rango",
            "norm":      None,
            "pass":      shape_ok,
            "icon":      "✓" if shape_ok else "✗",
        },
    ]
    fails = [ind["name"] for ind in indicators if not ind["pass"]]
    certifiable = len(fails) == 0

    if certifiable:
        if intact >= 99:
            level = "Excelente - Apto para certificacion INIAF"
        elif intact >= 98:
            level = "Bueno - Apto para certificacion INIAF"
        else:
            level = "Aceptable - Cumple norma INIAF"
    else:
        level = "No certificable - No cumple norma INIAF 2022"

    return indicators, certifiable, level, fails


def analyze_per_seed(file_bytes_list, n_reject_ext=0, n_total_ext=0):
    """
    Analiza cada semilla con la CNN y aplica criterios INIAF 2022.

    Args:
        file_bytes_list: lista de bytes de imagen.
        n_reject_ext: objetos no-soya detectados por el modulo de vision clasica.
        n_total_ext: total de objetos detectados (soya + no-soya).

    Returns dict con distribution, iniaf_indicators, certifiable, level, reasons.
    """
    try:
        model, idx2class = _get_model()
    except Exception as e:
        return {"error": f"No se pudo cargar el modelo CNN: {e}"}

    all_seeds = []
    for img_idx, file_bytes in enumerate(file_bytes_list):
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None: continue
        for seed_idx, r in enumerate(_detect_and_classify(img, model, idx2class)):
            r["image_index"] = img_idx
            r["seed_index"]  = seed_idx
            all_seeds.append(r)

    if not all_seeds:
        return {
            "total_seeds": 0, "distribution": [], "iniaf_indicators": [],
            "certifiable": False,
            "certification_level": "Sin datos - no se detectaron semillas",
            "certification_reasons": ["No se detectaron semillas en las imagenes."],
            "per_seed": [],
        }

    from collections import Counter
    total_cnn = len(all_seeds)
    counts    = Counter(s["class"] for s in all_seeds)

    # Para materia inerte usamos los datos externos si existen,
    # si no, la calculamos como 0 (solo se analizan las soya detectadas)
    grand_total = n_total_ext if n_total_ext > 0 else total_cnn
    pct_inerte  = (n_reject_ext / grand_total * 100) if grand_total > 0 else 0.0

    # Distribución porcentual (sobre total_cnn, base de la CNN)
    all_classes = list(CLASS_COLORS.keys())
    distribution = []
    pcts = {}
    for cls in all_classes:
        cnt = counts.get(cls, 0)
        pct = round(cnt / total_cnn * 100, 1) if total_cnn > 0 else 0.0
        pcts[cls] = pct
        distribution.append({
            "class":      cls,
            "label_es":   CLASS_LABELS_ES.get(cls, cls),
            "count":      cnt,
            "percentage": pct,
            "color":      CLASS_COLORS.get(cls, "#888"),
        })

    indicators, certifiable, level, fails = _build_iniaf_indicators(pcts, pct_inerte)

    return {
        "total_seeds":           total_cnn,
        "grand_total":           grand_total,
        "distribution":          distribution,
        "iniaf_indicators":      indicators,
        "certifiable":           certifiable,
        "certification_level":   level,
        "certification_reasons": fails,
        "per_seed":              all_seeds,
    }