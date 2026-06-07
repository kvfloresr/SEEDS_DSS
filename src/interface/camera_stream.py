"""
camera_stream.py — Transmite el video de la cámara YA PROCESADO (con las cajas y el
conteo dibujados) como un stream MJPEG, para mostrarlo dentro del sistema web.

Reutiliza la misma lógica de detección/conteo de src/live_camera.py, así que el
navegador ve exactamente lo mismo que la ventana de escritorio, pero embebido en la
página. El endpoint que lo expone está en src/interface/api.py (/api/camera/stream).
"""

import cv2

from src.live_camera import open_camera, process_frame, DEFAULTS

# Último conteo y último cuadro CRUDO (sin dibujos), para el frontend.
latest = {"n_soy": 0, "n_reject": 0, "n_total": 0}
latest_jpeg = {"data": None}     # bytes JPEG del último cuadro limpio (para capturar)


def gen_frames(camera=0, backend="msmf", width=1280, height=720):
    """Generador que abre la cámara, procesa cada cuadro y lo entrega como JPEG
    en formato multipart (MJPEG). Libera la cámara al cortarse la conexión."""
    cap = open_camera(camera, width, height, backend)
    if not cap.isOpened():
        # Cuadro de aviso si no se pudo abrir la cámara
        import numpy as np
        msg = np.full((360, 640, 3), 40, dtype="uint8")
        cv2.putText(msg, "No se pudo abrir la camara", (40, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 240), 2)
        ok, buf = cv2.imencode(".jpg", msg)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        return

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
    ratio = (real_w * real_h) / (640.0 * 480.0)
    params = {
        "min_area": int(DEFAULTS["min_area"] * ratio),
        "max_area": int(DEFAULTS["max_area"] * ratio),
        "thresh": 0,
        "invert": False,
        "color_sat": 45,
    }

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Guardar el cuadro CRUDO (limpio) para la captura/snapshot
            ok_raw, raw_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok_raw:
                latest_jpeg["data"] = raw_buf.tobytes()

            annotated, _, (n_soy, n_reject) = process_frame(frame, params)
            latest["n_soy"] = n_soy
            latest["n_reject"] = n_reject
            latest["n_total"] = n_soy + n_reject
            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + buf.tobytes() + b"\r\n")
    finally:
        cap.release()