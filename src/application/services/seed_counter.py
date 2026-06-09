import base64
import cv2
import numpy as np

from src.live_camera import process_frame, DEFAULTS


def count_from_bytes(file_bytes, params=None):
    """Cuenta a partir de los bytes de una imagen (lo que llega del frontend).
    Devuelve n_soy, n_reject, n_total y la imagen anotada en base64."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "imagen invalida"}

    h, w = img.shape[:2]
    ratio = (w * h) / (640.0 * 480.0)
    p = {
        "min_area": int(DEFAULTS["min_area"] * ratio),
        "max_area": int(DEFAULTS["max_area"] * ratio),
        "thresh": 0,
        "invert": False,
        "color_sat": 45,
    }
    if params:
        p.update(params)

    annotated, _, (n_soy, n_reject) = process_frame(img, p)

    overlay_b64 = None
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ok:
        overlay_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")

    return {
        "n_soy": n_soy,
        "n_reject": n_reject,
        "n_total": n_soy + n_reject,
        "overlay_b64": overlay_b64,
    }