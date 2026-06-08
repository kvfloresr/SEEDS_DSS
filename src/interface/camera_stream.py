"""
camera_stream.py — Cámara en un hilo de fondo dedicado.

PROBLEMA QUE RESUELVE: MSMF (el backend de cámara de Windows) solo funciona
en el hilo principal. Cuando Flask sirve el stream en un hilo secundario, MSMF
falla con errores -1072873822. Solución: la cámara corre en UN hilo de fondo
permanente; Flask solo lee el buffer de ese hilo.

VENTAJA ADICIONAL: la cámara se pre-inicia al arrancar el backend, así cuando
el usuario llega al paso 4 ya está lista (sin espera).

NOTAS DE ESTA VERSIÓN:
- Usa la webcam USB (índice 1 por defecto), no la cámara integrada (índice 0).
    Cámbialo con la variable de entorno CAM_INDEX si hiciera falta.
- Fuerza el códec MJPG antes de fijar la resolución: sin esto muchas webcams
    USB usan YUY2 (sin comprimir), no sostienen 720p y devuelven frames negros
    o a ~5 FPS (lag / entrecortado).
- Warm-up al abrir: descarta los primeros frames negros mientras el sensor
    ajusta la exposición, para que el stream NO empiece en negro.
- BUFFERSIZE=1 reduce la latencia (no acumula frames viejos).
- Procesa la detección cada 2 frames y re-codifica el snapshot solo de vez en
    cuando, para no saturar la CPU.
"""

import os
import platform
import threading
import time

import cv2
import numpy as np

from src.live_camera import process_frame, DEFAULTS

# ── Configuración de cámara ──────────────────────────────────────────────────── #
# La webcam USB suele ser el índice 1 (la integrada de la laptop es 0).
# Se puede sobreescribir sin tocar código:  set CAM_INDEX=2  (Windows)
CAM_INDEX = int(os.environ.get("CAM_INDEX", "1"))

# ── Estado compartido ──────────────────────────────────────────────────────── #
_lock          = threading.Lock()
_annotated_jpg = None          # JPEG anotado (detección + HUD) para el stream
_running       = False
_thread        = None
_cam_ready     = threading.Event()

latest      = {"n_soy": 0, "n_reject": 0, "n_total": 0}
latest_jpeg = {"data": None}   # JPEG limpio (sin dibujos) para snapshot


# ── Apertura robusta: prueba varios backends hasta que funcione ────────────── #
def _open_camera(index=CAM_INDEX, width=1280, height=720):
    """Intenta abrir la cámara con varios backends. Devuelve (cap, real_w, real_h)."""
    if platform.system() == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    for backend in backends:
        try:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release(); continue

            # 1) Forzar MJPG ANTES de la resolución. Sin esto la cámara usa
            #    YUY2 (sin comprimir) y no sostiene 720p -> negro / lag.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # menos latencia / entrecortado

            # 2) Warm-up: descarta frames negros mientras el sensor ajusta
            #    exposición. Solo aceptamos cuando el frame ya tiene luz.
            warm_ok = False
            for _ in range(15):
                ok, fr = cap.read()
                if ok and fr is not None and fr.mean() > 5:   # no es negro
                    warm_ok = True
                    break
                time.sleep(0.1)

            if warm_ok:
                rw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                rh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                bname = {cv2.CAP_DSHOW: "DSHOW", cv2.CAP_MSMF: "MSMF",
                        cv2.CAP_ANY: "ANY"}.get(backend, str(backend))
                print(f"[CAM] Cámara {index} abierta con {bname} — {rw}x{rh}")
                return cap, rw, rh

            print(f"[CAM] Índice {index}: solo frames negros con este backend, probando otro...")
            cap.release()
        except Exception as e:
            print(f"[CAM] Backend {backend} falló: {e}")
    return None, 0, 0


# ── Hilo de captura ─────────────────────────────────────────────────────────── #
def _capture_loop(cam_index=CAM_INDEX, width=1280, height=720):
    global _annotated_jpg, _running

    cap, rw, rh = _open_camera(cam_index, width, height)
    if cap is None:
        print(f"[CAM] ERROR: no se pudo abrir la cámara {cam_index}")
        _running = False
        _cam_ready.set()
        return

    ratio  = (rw * rh) / (640.0 * 480.0)
    params = {
        "min_area":  int(DEFAULTS["min_area"] * ratio),
        "max_area":  int(DEFAULTS["max_area"] * ratio),
        "thresh": 0, "invert": False, "color_sat": 45,
    }
    _cam_ready.set()
    print("[CAM] Lista — iniciando stream de frames.")

    fails = 0
    frame_idx = 0
    while _running:
        ok, frame = cap.read()
        if not ok:
            fails += 1
            if fails > 40:
                print("[CAM] Demasiados fallos consecutivos, reintentando cámara...")
                cap.release()
                time.sleep(2)
                cap, rw, rh = _open_camera(cam_index, width, height)
                if cap is None:
                    print("[CAM] No se pudo reconectar la cámara.")
                    break
                fails = 0
            time.sleep(0.05)
            continue
        fails = 0
        frame_idx += 1

        # ── Frame limpio (para snapshot): no hace falta en CADA frame ──
        #    Lo guardamos en el primer frame y luego cada 6 para no gastar
        #    CPU codificando dos JPEG por frame.
        if frame_idx == 1 or frame_idx % 6 == 0:
            ok_r, raw = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok_r:
                with _lock:
                    latest_jpeg["data"] = raw.tobytes()

        # ── Frame anotado (para stream): detección cada 2 frames ──
        #    La detección clásica + dibujo es lo más caro; a ~15 FPS anotados
        #    el video sigue fluido y la CPU respira.
        if frame_idx % 2 == 0:
            try:
                annotated, _, (ns, nr) = process_frame(frame, params)
                ok_a, ann = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok_a:
                    with _lock:
                        _annotated_jpg = ann.tobytes()
                        latest["n_soy"]    = ns
                        latest["n_reject"] = nr
                        latest["n_total"]  = ns + nr
            except Exception as e:
                print(f"[CAM] Error procesando frame: {e}")

        time.sleep(0.01)   # deja respirar al resto del proceso

    cap.release()
    print("[CAM] Cámara liberada.")


# ── API pública ──────────────────────────────────────────────────────────────── #
def start_camera(cam_index=CAM_INDEX, width=1280, height=720):
    """Arranca el hilo de captura (una sola vez). Seguro llamarlo varias veces."""
    global _thread, _running, _cam_ready
    if _running and _thread and _thread.is_alive():
        return
    _running   = True
    _cam_ready = threading.Event()
    _thread = threading.Thread(
        target=_capture_loop,
        args=(cam_index, width, height),
        daemon=True, name="CameraCapture"
    )
    _thread.start()
    print(f"[CAM] Hilo de captura iniciado (cámara {cam_index}).")


def gen_frames(camera=CAM_INDEX, backend="auto", width=1280, height=720):
    """Generador MJPEG para Flask Response. Lee del buffer del hilo de fondo."""
    # Asegurar que el hilo esté corriendo
    if not _running or _thread is None or not _thread.is_alive():
        start_camera(camera, width, height)
    _cam_ready.wait(timeout=20)   # Espera hasta 20 s a que la cámara abra

    while True:
        with _lock:
            fb = _annotated_jpg
        if fb is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + fb + b"\r\n")
        time.sleep(0.04)    # ~25 FPS al cliente