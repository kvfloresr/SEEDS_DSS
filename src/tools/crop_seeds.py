import argparse
from pathlib import Path

import cv2

from src.live_camera import (
    segment, _circularity,
    DEFAULTS, CIRC_MIN, SIZE_LOW, SIZE_HIGH,
)
# Reutilizamos el MISMO recorte que la app en inferencia -> consistencia total.
from src.application.services.per_seed_analysis import _crop_seed


def list_images(folder, exts=(".bmp", ".jpg", ".jpeg", ".png")):
    files = []
    for e in exts:
        files += list(folder.glob(f"*{e}"))
        files += list(folder.glob(f"*{e.upper()}"))
    return sorted(set(files))


def crop_seeds_from_image(img_bgr):
    """Misma lógica de selección que _detect_and_classify, pero en vez de
    predecir, devuelve los recortes (ya 128x128) para guardarlos."""
    h, w = img_bgr.shape[:2]
    ratio = (w * h) / (640.0 * 480.0)
    min_area = int(DEFAULTS["min_area"] * ratio)
    max_area = int(DEFAULTS["max_area"] * ratio)

    mask = segment(img_bgr, 0, False)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cands = [(c, cv2.contourArea(c)) for c in contours
            if min_area <= cv2.contourArea(c) <= max_area]
    if not cands:
        return []

    med = sorted(a for _, a in cands)[len(cands) // 2]   # tamaño típico de semilla

    crops = []
    for cnt, area in cands:
        if _circularity(cnt, area) < CIRC_MIN:
            continue                                      # forma irregular = ruido
        if med > 0 and (area < SIZE_LOW * med or area > SIZE_HIGH * med):
            continue                                      # demasiado chica/grande (p.ej. 2 pegadas)
        x, y, bw, bh = cv2.boundingRect(cnt)
        crop = _crop_seed(img_bgr, cnt, x, y, bw, bh)     # cuadrado centrado + resize 128, IDÉNTICO a inferencia
        if crop is not None:
            crops.append(crop)
    return crops


def main():
    ap = argparse.ArgumentParser(description="Recorta semillas individuales de fotos multi-semilla.")
    ap.add_argument("--input", default="data/raw_multi",
                    help="Carpeta con subcarpetas por clase (fotos de varias semillas).")
    ap.add_argument("--output", default="data/raw",
                    help="Carpeta destino (subcarpetas por clase con los recortes 128x128).")
    ap.add_argument("--min-per-photo", type=int, default=3,
                    help="Avisa si una foto produce menos recortes que esto (luz/fondo malos).")
    args = ap.parse_args()

    in_root, out_root = Path(args.input), Path(args.output)
    if not in_root.exists():
        print(f"[ERROR] No existe la carpeta de entrada: {in_root}")
        return

    classes = [d for d in in_root.iterdir() if d.is_dir()]
    if not classes:
        print(f"[ERROR] No hay subcarpetas de clase dentro de {in_root}")
        return

    total = 0
    for cls_dir in sorted(classes):
        cls = cls_dir.name
        out_dir = out_root / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        photos = list_images(cls_dir)
        print(f"\n[{cls}] {len(photos)} fotos de entrada")
        n_cls = 0
        for ph in photos:
            img = cv2.imread(str(ph))
            if img is None:
                print(f"  [skip] no se pudo leer {ph.name}")
                continue
            crops = crop_seeds_from_image(img)
            if len(crops) < args.min_per_photo:
                print(f"  [aviso] {ph.name}: solo {len(crops)} recortes "
                    f"(revisa iluminación / fondo / enfoque)")
            for i, c in enumerate(crops):
                out_name = f"{ph.stem}_seed{i:03d}.png"
                cv2.imwrite(str(out_dir / out_name), c)
                n_cls += 1
        print(f"  -> {n_cls} recortes guardados en {out_dir}")
        total += n_cls

    print(f"\n[OK] Total de recortes generados: {total}")
    print("Revisa visualmente algunas carpetas y borra recortes obviamente malos")
    print("(dos semillas pegadas, sombras, basura) antes de hacer el split + entrenar.")


if __name__ == "__main__":
    main()