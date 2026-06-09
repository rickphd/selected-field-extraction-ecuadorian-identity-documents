from rknnlite.api import RKNNLite 
import cv2
import numpy as np
from ultralytics import YOLO
from text_converter import TextConverter
import os
from PIL import Image
import random

model_ocr = RKNNLite()
model_segrot = RKNNLite()
model_ocr.load_rknn('models/model_ocr.rknn')
model_segrot.load_rknn('models/model_segrot.rknn')
model_ocr.init_runtime()
model_segrot.init_runtime()

def preproces(image, size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def seg_rot(image):
    img_size = image.size
    img_trans = preproces(image, (224, 224))
    # Inferencia
    results = model_segrot.inference(inputs=[img_trans])
    mask , angle = results
    angle = np.squeeze(angle)
    mask = np.squeeze(mask)
    mask = cv2.resize(mask, (img_size), interpolation=cv2.INTER_LINEAR)
    mask = (mask > 0.5).astype(np.uint8) * 255
    angle = np.argmax(angle)
    angle = angle * 5
    #print("Ángulo predicho (grados):", angle)
    img = rotate_image(np.array(image), -angle)
    #predicted_mask.save(f"uploads/mask/mascara_predicha_{random_id}.jpg")  # Guardar la máscara para depuración
    predicted_mask = rotate_image(mask, -angle)
    img_np = np.array(img)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_with_rect = img_np.copy()
    warped = None
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(img_with_rect, [box], 0, (255, 0, 0), 3)
    
        # Puntos fuente
        src_pts = box.astype("float32")
    
        # Ordenar consistentemente los puntos: [top-left, top-right, bottom-right, bottom-left]
        rect_pts = np.zeros((4, 2), dtype="float32")
        s = src_pts.sum(axis=1)
        rect_pts[0] = src_pts[np.argmin(s)]  # top-left
        rect_pts[2] = src_pts[np.argmax(s)]  # bottom-right
    
        diff = np.diff(src_pts, axis=1)
        rect_pts[1] = src_pts[np.argmin(diff)]  # top-right
        rect_pts[3] = src_pts[np.argmax(diff)]  # bottom-left
    
        # Calcular dimensiones en base a distancias
        W = int(np.linalg.norm(rect_pts[0] - rect_pts[1]))
        H = int(np.linalg.norm(rect_pts[1] - rect_pts[2]))
    
        if W > 0 and H > 0:
            dst_pts = np.array([
                [0, 0],
                [W-1, 0],
                [W-1, H-1],
                [0, H-1]
            ], dtype="float32")
    
            # Transformación de perspectiva
            M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
            warped = cv2.warpPerspective(img_np, M, (W, H))
            #cv2.imwrite(f"uploads/recortes/cedula_recortada_{random_id}.jpg", warped)  # Guardar la imagen recortada para depuración
    return warped

def order_points(pts):
    """Ordenar los puntos del rectángulo en orden:
    topleft, topright, bottomright, bottomleft
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def procesar_resultado(result, etiquetas, salida="uploads/recortes_txt", idx=0):
    """
    Procesa un solo 'result' del YOLO, guarda los recortes y los devuelve en memoria.

    Parámetros:
    - result: salida de YOLO (ej: r de 'for r in results')
    - etiquetas: diccionario {cls_id: nombre_clase}
    - salida: carpeta donde se guardan los recortes
    - idx: índice opcional para nombrar archivos

    Retorna:
    - lista de diccionarios con {"array": recorte_numpy, "path": ruta_archivo, "tipo": clase}
    """
    os.makedirs(salida, exist_ok=True)
    img = result.orig_img
    boxes = result.obb  # cajas orientadas
    recortes = []

    for j, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        if cls_id in etiquetas:
            
            # Extraer los 4 puntos del OBB
            pts = box.xyxyxyxy[0].cpu().numpy().reshape(4, 2)
            rect = order_points(pts)

            # Calcular dimensiones
            (tl, tr, br, bl) = rect
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)

            maxWidth = int(max(widthA, widthB))
            maxHeight = int(max(heightA, heightB))

            # --- Forzar lado largo horizontal ---
            if maxHeight > maxWidth:
                maxWidth, maxHeight = maxHeight, maxWidth
                dst = np.array(
                    [
                        [0, 0],
                        [0, maxHeight - 1],
                        [maxWidth - 1, maxHeight - 1],
                        [maxWidth - 1, 0],
                    ],
                    dtype="float32",
                )
            else:
                dst = np.array(
                    [
                        [0, 0],
                        [maxWidth - 1, 0],
                        [maxWidth - 1, maxHeight - 1],
                        [0, maxHeight - 1],
                    ],
                    dtype="float32",
                )

            # Warp de perspectiva
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

            # Guardar recorte
            tipo = etiquetas[cls_id]
            #random_id = random.randint(0, 9999999)
            #out_path = os.path.join(salida, f"{random_id}_{j}_{tipo}.jpg")
            #print(f"Guardado: {out_path}")
            #cv2.imwrite(out_path, warped)

            recortes.append({
                "array": warped,
                #"path": out_path,
                "tipo": tipo
            })

    return recortes

def procesar_results(results, etiquetas, salida="uploads/recortes_txt"):
    """
    Procesa directamente la lista 'results' de YOLO.

    Retorna:
    - lista con todos los recortes de todos los results
    """
    all_recortes = []
    for i, result in enumerate(results):
        recs = procesar_resultado(result, etiquetas, salida=salida, idx=i)
        #print(recs)
        all_recortes.extend(recs)
    return all_recortes

def extraer_rostro(image, model, device='cpu'):
    results = model.predict(image, conf=0.75, device=device, verbose=False)
    
    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xywh.cpu().numpy()
        annotated_frame = results[0].orig_img.copy()
        rostros = []
        
        for box in boxes:
            x, y, width, height = box
            width, height = width * 1.6, height * 1.6
            x1, y1 = int(x - width / 2), int(y - height / 2)
            x2, y2 = int(x + width / 2), int(y + height / 2)

            # Limitar coordenadas a los bordes de la imagen
            h, w = annotated_frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            rostro = annotated_frame[y1:y2, x1:x2]
            if rostro is not None and rostro.size > 0:
                return rostro
        return None
    else:
        return None
