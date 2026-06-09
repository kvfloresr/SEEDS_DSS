import argparse
import platform
import sys
import time

import cv2
import numpy as np


BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}


def open_camera(index, width=1280, height=720, backend="dshow"):
    """Abre la cámara y le PIDE la resolución deseada. En Windows el 'motor'
    importa: DirectShow (dshow) y Media Foundation (msmf) numeran las cámaras
    distinto, así que se puede elegir. Fuera de Windows se ignora el motor."""
    if platform.system() == "Windows":
        flag = BACKENDS.get(backend, cv2.CAP_ANY)
        cap = cv2.VideoCapture(index, flag)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(index)          # último recurso
    else:
        cap = cv2.VideoCapture(index)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

# --------------------------- Configuración ---------------------------------- #
DEFAULTS = {"min_area": 250, "max_area": 80000, "thresh": 0}

# "No soya" se decide por TAMAÑO atípico respecto a las demás semillas de la escena
# (más robusto que el color en semillas pálidas). Tolerancia respecto a la mediana:
SIZE_LOW = 0.45     # menor a 0.45x la mediana -> impureza pequeña
SIZE_HIGH = 2.2     # mayor a 2.2x la mediana  -> objeto grande / 2 semillas pegadas
CIRC_MIN = 0.20     # forma muy irregular (palito, hoja) -> no es semilla

# ...y también por COLOR claramente "antinatural" (rosa/magenta/lila/azul), que la
# soya no tiene. En esos colores el VERDE se hunde por debajo del AZUL; en toda la
# soya (tostada, roja, rosada-café) el verde queda por encima del azul.
COOL_MARGIN = 15    # azul supera al verde por más de esto -> color no-soya
COLOR_SAT_GATE = 45 # saturación mínima para confiar en el color (no juzga grises pálidos)

# Paleta (BGR)
COL_BG      = (32, 30, 28)
COL_SOY     = (96, 201, 90)
COL_REJECT  = (74, 74, 240)
COL_TEXT    = (240, 240, 240)
COL_MUTED   = (170, 170, 170)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------- Helpers de dibujo ------------------------------ #
def rounded_rect(img, p1, p2, color, thickness=-1, radius=12):
    x1, y1 = p1
    x2, y2 = p2
    r = min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r),
                       (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, color, -1)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
        for cx, cy, a in ((x1 + r, y1 + r, 180), (x2 - r, y1 + r, 270),
                          (x1 + r, y2 - r, 90), (x2 - r, y2 - r, 0)):
            cv2.ellipse(img, (cx, cy), (r, r), a, 0, 90, color, thickness)


def panel(img, p1, p2, color=COL_BG, alpha=0.78, radius=14):
    overlay = img.copy()
    rounded_rect(overlay, p1, p2, color, -1, radius)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def corner_box(img, x, y, w, h, color, thickness=2, length=18):
    L = min(length, w // 2, h // 2)
    pts = [
        ((x, y), (x + L, y)), ((x, y), (x, y + L)),
        ((x + w, y), (x + w - L, y)), ((x + w, y), (x + w, y + L)),
        ((x, y + h), (x + L, y + h)), ((x, y + h), (x, y + h - L)),
        ((x + w, y + h), (x + w - L, y + h)), ((x + w, y + h), (x + w, y + h - L)),
    ]
    for a, b in pts:
        cv2.line(img, a, b, color, thickness, cv2.LINE_AA)


def chip(img, x, y, text, bg, fg=COL_TEXT, scale=0.5):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    pad = 6
    y_top = max(0, y - th - pad * 2)
    rounded_rect(img, (x, y_top), (x + tw + pad * 2, y_top + th + pad * 2), bg, -1, 6)
    cv2.putText(img, text, (x + pad, y_top + th + pad - 1),
                FONT, scale, fg, 1, cv2.LINE_AA)


# --------------------------- Visión clásica --------------------------------- #
def segment(frame_bgr, thresh_value, invert):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if thresh_value <= 0:
        _, mask = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY_INV)
    if invert:
        mask = cv2.bitwise_not(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _circularity(contour, area):
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return (4 * np.pi * area) / (perimeter ** 2)


def _is_cool_color(frame_bgr, hsv, contour, sat_gate):
    """True si el objeto tiene un color antinatural y vívido (rosa/magenta/lila/azul),
    que la soya no tiene. En esos colores el verde cae por debajo del azul. Mide el
    interior del contorno (erosiona para no tomar el borde ni el fondo) y no juzga
    objetos pálidos/grises (saturación baja)."""
    m = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(m, [contour], -1, 255, thickness=cv2.FILLED)
    er = cv2.erode(m, np.ones((5, 5), np.uint8))
    if cv2.countNonZero(er) > 30:
        m = er
    mean_b, mean_g, _r, _ = cv2.mean(frame_bgr, mask=m)
    mean_s = cv2.mean(hsv[:, :, 1], mask=m)[0]
    return mean_s >= sat_gate and (mean_b - mean_g) > COOL_MARGIN


def process_frame(frame_bgr, params):
    mask = segment(frame_bgr, params["thresh"], params["invert"])
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out = frame_bgr.copy()
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    s = max(0.85, frame_bgr.shape[1] / 960.0)   # escala de la interfaz
    sat_gate = params.get("color_sat", COLOR_SAT_GATE)

    # --- Paso 1: reunir candidatos dentro del rango de área ---
    cands = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < params["min_area"] or area > params["max_area"]:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cands.append((cnt, area, _circularity(cnt, area), x, y, w, h))

    # --- Paso 2: tamaño típico de semilla = mediana de las áreas ---
    areas = sorted(a for _, a, _, _, _, _, _ in cands)
    median = areas[len(areas) // 2] if areas else 0

    # --- Paso 3: clasificar cada candidato (tamaño + forma + color frío) ---
    n_soy, n_reject = 0, 0
    for cnt, area, circ, x, y, w, h in cands:
        es_soya = True
        if circ < CIRC_MIN:
            es_soya = False                          # forma muy irregular
        elif median > 0 and (area < SIZE_LOW * median or area > SIZE_HIGH * median):
            es_soya = False                          # tamaño atípico
        elif sat_gate > 0 and _is_cool_color(frame_bgr, hsv, cnt, sat_gate):
            es_soya = False                          # color frío (lila/azul) = no soya

        if es_soya:
            n_soy += 1
            corner_box(out, x, y, w, h, COL_SOY, max(2, int(2 * s)), int(18 * s))
            chip(out, x, y, str(n_soy), COL_SOY, (20, 60, 20), 0.5 * s)
        else:
            n_reject += 1
            corner_box(out, x, y, w, h, COL_REJECT, max(2, int(2 * s)), int(18 * s))
            chip(out, x, y, "no soya", COL_REJECT, COL_TEXT, 0.5 * s)

    _draw_hud(out, n_soy, n_reject, s)
    return out, mask, (n_soy, n_reject)


def _draw_hud(img, n_soy, n_reject, s=1.0):
    h, w = img.shape[:2]

    def S(v):  # escalar un valor
        return int(v * s)

    # Tarjeta de conteo (arriba a la izquierda)
    panel(img, (S(16), S(16)), (S(16) + S(234), S(16) + S(116)), radius=S(14))
    cv2.putText(img, "SEEDS DSS", (S(32), S(44)), FONT, 0.5 * s, COL_MUTED,
                max(1, int(s)), cv2.LINE_AA)
    cv2.putText(img, str(n_soy), (S(30), S(112)), FONT, 1.7 * s, COL_SOY,
                max(2, int(3 * s)), cv2.LINE_AA)
    cv2.putText(img, "semillas de soya", (S(118), S(88)), FONT, 0.45 * s, COL_TEXT,
                max(1, int(s)), cv2.LINE_AA)
    cv2.circle(img, (S(126), S(102)), max(4, S(5)), COL_REJECT, -1, cv2.LINE_AA)
    cv2.putText(img, f"{n_reject} no soya", (S(140), S(107)), FONT, 0.45 * s, COL_MUTED,
                max(1, int(s)), cv2.LINE_AA)

    # Barra de controles (abajo)
    panel(img, (S(16), h - S(46)), (S(16) + S(450), h - S(14)), alpha=0.6, radius=S(10))
    cv2.putText(img, "q salir   .   s captura   .   i invertir   .   h mascara",
                (S(34), h - S(24)), FONT, 0.45 * s, COL_MUTED, max(1, int(s)), cv2.LINE_AA)


# --------------------------- Controles -------------------------------------- #
def setup_controls():
    cv2.namedWindow("Controles", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controles", 420, 160)
    cv2.createTrackbar("min_area", "Controles", DEFAULTS["min_area"], 60000, lambda v: None)
    cv2.createTrackbar("max_area", "Controles", DEFAULTS["max_area"], 800000, lambda v: None)
    cv2.createTrackbar("thresh(0=auto)", "Controles", DEFAULTS["thresh"], 255, lambda v: None)
    cv2.createTrackbar("color_sat(0=off)", "Controles", COLOR_SAT_GATE, 200, lambda v: None)


def read_controls():
    return {
        "min_area": cv2.getTrackbarPos("min_area", "Controles"),
        "max_area": max(cv2.getTrackbarPos("max_area", "Controles"), 1),
        "thresh": cv2.getTrackbarPos("thresh(0=auto)", "Controles"),
        "color_sat": cv2.getTrackbarPos("color_sat(0=off)", "Controles"),
    }


# --------------------------- Bucle principal -------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Conteo de semillas de soya en vivo.")
    ap.add_argument("--camera", type=int, default=0, help="Índice de la cámara.")
    ap.add_argument("--backend", choices=["dshow", "msmf", "any"], default="msmf",
                    help="Motor de cámara en Windows (dshow/msmf/any).")
    ap.add_argument("--width", type=int, default=1280, help="Ancho deseado (def. 1280).")
    ap.add_argument("--height", type=int, default=720, help="Alto deseado (def. 720).")
    ap.add_argument("--display", type=int, default=900,
                    help="Ancho de la VENTANA en pantalla (no afecta la captura).")
    args = ap.parse_args()

    cap = open_camera(args.camera, args.width, args.height, args.backend)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara {args.camera}.")
        print("        Prueba con otro índice: --camera 1  (o 0, 2).")
        print("        Y cierra cualquier app que use la cámara (Zoom, navegador, etc.).")
        sys.exit(1)

    # Resolución REAL que entregó la cámara (puede diferir de la pedida)
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] Cámara abierta a {real_w}x{real_h}. Presiona 'q' para salir.")

    setup_controls()
    # Ajustar el área por defecto según la resolución (más píxeles -> semilla más grande)
    ratio = (real_w * real_h) / (640.0 * 480.0)
    cv2.setTrackbarPos("min_area", "Controles", int(DEFAULTS["min_area"] * ratio))
    cv2.setTrackbarPos("max_area", "Controles", int(DEFAULTS["max_area"] * ratio))

    invert = False
    show_mask = False

    # Ventana redimensionable, a un tamaño cómodo en pantalla (manteniendo proporción).
    # La captura sigue a resolución completa; solo se muestra más pequeña.
    win = "SEEDS DSS - Camara en vivo"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    disp_w = min(args.display, real_w)
    disp_h = int(real_h * disp_w / real_w)
    cv2.resizeWindow(win, disp_w, disp_h)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No se recibió cuadro de la cámara.")
            break
        params = read_controls()
        params["invert"] = invert
        annotated, mask, _ = process_frame(frame, params)
        cv2.imshow(win, annotated)
        if show_mask:
            cv2.imshow("Mascara (depuracion)", mask)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = f"captura_{int(time.time())}.jpg"
            cv2.imwrite(fname, annotated)
            print(f"[SNAP] Captura guardada: {fname}")
        elif key == ord("i"):
            invert = not invert
        elif key == ord("h"):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow("Mascara (depuracion)")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()