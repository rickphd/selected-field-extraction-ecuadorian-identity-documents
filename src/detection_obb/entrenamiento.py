from ultralytics import YOLO
import torch

if __name__ == "__main__":
    models=YOLO('yolo11n-obb.pt')  # load a pretrained model (recommended for training)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    models.train(
        data="data.yaml",  # Ruta al archivo de configuración
        epochs=250,  # Número de épocas
        imgsz=640,  # Tamaño de entrada de imagen
        batch=16,  # Tamaño del batch
        workers=8,  # Núm. de workers para carga de datos
        device=[-1,-1],  # Usa la GPU 0; usa "cpu" para CPU
        name="yolo11_obb_custom",  # Nombre de la carpeta del experimento
        verbose=True,
        patience=10,
        lr0=1e-3,
        lrf=0.001,
        degrees=5,
        optimizer="AdamW",
        # weight_decay=1e-6,
        hsv_v=0.5,
        hsv_s=0.75,
        flipud=0.25,
        fliplr=0.25,
        translate=0.5,
        scale=0.7,
        # shear=2,
        copy_paste_mode="mixup",
    )
