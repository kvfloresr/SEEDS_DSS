import cv2

def preprocess_image(img):
    """
    Preprocesamiento automático:
    - Normalización de iluminación
    - Eliminación de ruido (mediana + gaussiano)
    - Segmentación
    - Corrección geométrica
    """

    # Normalización de iluminación
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq,a,b))
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # Eliminación de ruido
    img_denoised = cv2.medianBlur(img_eq, 3)
    img_denoised = cv2.GaussianBlur(img_denoised, (3,3), 0)

    # Segmentación 
    gray = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Corrección geométrica 
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    seed_crop = img_denoised[y:y+h, x:x+w]
    seed_resized = cv2.resize(seed_crop, (128,128))

    return seed_resized, mask
