import cv2
import numpy as np

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

    # Color y uniformidad
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mean_color = np.mean(hsv[:, :, 0])
    std_color = np.std(hsv[:, :, 0])
    homogeneity = "Uniforme" if std_color < 20 else "Variable"

    # Impurezas (materia inerte)
    impurities_mask = cv2.inRange(hsv, (0, 0, 0), (179, 255, 50))
    impurity_ratio = np.sum(impurities_mask > 0) / (img.shape[0] * img.shape[1]) * 100

    # Daños mecánicos
    edges = cv2.Canny(gray, 100, 200)
    damage_ratio = np.sum(edges > 0) / area * 100
    damage_level = "Aceptable" if damage_ratio <= 1 else "Crítico"

    # Pureza física (según área limpia)
    pureza_fisica = max(98 - impurity_ratio, 0)

    features = {
        "Pureza física (%)": round(pureza_fisica, 2),
        "Materia inerte (%)": round(impurity_ratio, 2),
        "Daños mecánicos (%)": round(damage_ratio, 2),
        "Homogeneidad de color": homogeneity,
        "Forma y tamaño": f"Circularidad {circularity:.2f}, Tamaño relativo {size_ratio * 100:.1f}%",
        "Trazabilidad digital": "Completa"
    }

    return features, overlay