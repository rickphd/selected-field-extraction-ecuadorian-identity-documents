from model import MobilnetSegRot, ResNet50SegRot, EfficientNetSegRot
import torch 
import torch.nn as nn 
from torchvision.transforms import v2 
from utils import Load_Dataset
import torch.utils.tensorboard as tb 
import numpy as np
import datetime 
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torchmetrics import F1Score
import os
import gc


def clear_gpu_memory():
    """Limpia exhaustivamente la memoria GPU"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


def calculate_iou(predicted_mask, target_mask, epsilon=1e-6):
    """Calcula IoU entre máscaras predichas y reales"""
    predicted_mask = torch.sigmoid(predicted_mask)  # probabilidades
    predicted_mask = (predicted_mask > 0.5).float()  # matriz true/false -> 1/0
    
    predicted_mask = predicted_mask.view(-1).bool()  # permite hacer operacion & y |
    target_mask = target_mask.view(-1).bool()
    
    intersection = (predicted_mask & target_mask).float().sum()
    union = (predicted_mask | target_mask).float().sum()
    
    iou = (intersection + epsilon) / (union + epsilon)
    return iou.item()


def save_model(model, name):
    """Guarda el estado del modelo"""
    torch.save(model.state_dict(), name)


def train_model(model, model_name, train_dataloader, val_dataloader, device, num_classes):
    """Entrena un modelo específico"""
    print(f"\n{'='*60}")
    print(f"🚀 ENTRENANDO MODELO: {model_name}")
    print(f"{'='*60}")
    
    # Limpiar memoria antes de comenzar
    clear_gpu_memory()
    
    # Crear TensorBoard writer específico para este modelo
    model_log_dir = Path("runs") / f'{model_name}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    model_log_dir.mkdir(parents=True, exist_ok=True)
    writer = tb.SummaryWriter(model_log_dir)
    print(f"📝 Logs para {model_name}: {model_log_dir}")
    
    # Criterios de pérdida y métricas
    criterion_mask = nn.BCEWithLogitsLoss()
    criterion_rot = nn.CrossEntropyLoss()
    f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
    
    # Optimizador y scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=3,
        factor=0.1,
    )
    
    # Configuración de entrenamiento
    best_metric = 0
    patience = 5
    patience_counter = 0
    num_epochs = 300
    add_graf = True
    
    # Intentar cargar modelo previo
    model_path = f"{model_name}.pth"
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Modelo {model_name} cargado desde {model_path}")
    else:    
        print(f"❌ No se encontró {model_path}, entrenando desde cero.")
    
    model.to(device)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # ======= TRAIN =======
        model.train()
        running_loss = 0
        train_iou_list = []
        f1_train = 0
        
        for img, mask, clase in tqdm(train_dataloader, desc="Training", colour="green"):
            img = img.to(device).float()
            mask = mask.to(device).float()
            clase = clase.to(device).long()
            
            pred_mask, pred_clas = model(img)
            
            # Calcular métricas
            iou = calculate_iou(pred_mask, mask)
            loss_mask = criterion_mask(pred_mask, mask)
            loss_rot = criterion_rot(pred_clas, clase)
            train_iou_list.append(iou)
            f1_train += f1(pred_clas, clase).item()
            total_loss = loss_mask + loss_rot
            
            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
        
        train_loss = running_loss / len(train_dataloader)
        f1_train /= len(train_dataloader)
        train_iou = np.mean(train_iou_list)
        
        # Log training metrics
        if writer:
            writer.add_scalar("f1/train", f1_train, epoch)
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("mean_iou/train", train_iou, epoch)
            writer.add_images("images/train_input", img[:6].cpu(), epoch)
            writer.add_images("images/train_label", mask[:6].cpu(), epoch)
            writer.add_images("images/train_pred", torch.sigmoid(pred_mask[:6]).cpu(), epoch)
        
        print(f"[Train] Loss: {train_loss:.4f} | IoU: {train_iou*100:.2f}% | F1: {f1_train*100:.2f}%")
        
        # ======= VALIDATION =======
        model.eval()
        val_loss = 0
        f1_val = 0
        val_iou_list = []
        
        with torch.no_grad():
            for img, mask, clase in tqdm(val_dataloader, desc="Validation"):
                img = img.to(device).float()
                mask = mask.to(device).float()
                clase = clase.to(device).long()
                
                pred_mask, pred_clas = model(img)
                
                # Calcular métricas
                iou = calculate_iou(pred_mask, mask)
                val_iou_list.append(iou)
                loss_mask = criterion_mask(pred_mask, mask)
                loss_rot = criterion_rot(pred_clas, clase)
                f1_val += f1(pred_clas, clase).item()
                
                total_loss = loss_mask + loss_rot
                val_loss += total_loss.item()
        
        val_loss /= len(val_dataloader)
        f1_val /= len(val_dataloader)
        val_iou = np.mean(val_iou_list)
        
        # Scheduler
        scheduler.step(val_loss)
        
        # Log validation metrics
        if writer:
            writer.add_scalar("hyperparams/learning_rate", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("mean_iou/val", val_iou, epoch)
            writer.add_scalar("f1/val", f1_val, epoch)
            writer.add_images("images/val_input", img[:6].cpu(), epoch)
            writer.add_images("images/val_label", mask[:6].cpu(), epoch)
            writer.add_images("images/val_pred", torch.sigmoid(pred_mask[:6]).cpu(), epoch)
        
        print(f"[Val] Loss: {val_loss:.4f} | IoU: {val_iou*100:.2f}% | F1: {f1_val*100:.2f}%")
        
        # Agregar gráfico del modelo (solo una vez)
        if add_graf:
            writer.add_graph(model, img)
            add_graf = False
       
        # Early Stopping y guardado del mejor modelo
        if f1_val > best_metric:
            best_metric = f1_val
            patience_counter = 0
            save_model(model, model_path)
            print(f"✅ Mejor modelo {model_name} guardado. F1: {best_metric:.4f}")
        else:
            if optimizer.param_groups[0]["lr"] < 2e-8:
                patience_counter += 1
                print(f"🔻 Reducción de LR {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"🛑 Early stopping para {model_name}")
                break
     
    # Limpiar cache CUDA al final de cada Modelo
    clear_gpu_memory()
        
    # Cerrar el writer específico del modelo
    writer.close()
    print(f"✅ Entrenamiento de {model_name} completado. Mejor F1: {best_metric:.4f}")
    print(f"📝 Logs guardados en: {model_log_dir}")
    return best_metric


def main():
    """Función principal para entrenar todos los modelos secuencialmente"""
    print("🎯 ENTRENAMIENTO SECUENCIAL DE MODELOS SegRot")
    print("="*60)
    
    # ======================== CONFIGURACIÓN ========================
    torch.manual_seed(17)
    df_path = Path("../Data_df/progreso_orientacion_aument.csv").resolve()
    path = Path("../Data_img/imagenes_cedula_peq").resolve()
    
    # Verificar que los archivos existan
    if not df_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo CSV: {df_path}")
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el directorio de imágenes: {path}")
    
    df = pd.read_csv(df_path).drop(columns=["Unnamed: 0"], errors="ignore")
    uniques = np.unique(df['Clase'].values)
    num_classes = len(uniques)
    
    print(f"📊 Dataset cargado: {len(df)} muestras, {num_classes} clases")
    print(f"🏷️ Clases: {uniques}")
    
    # Crear directorio base para los logs
    base_log_dir = Path("runs")
    base_log_dir.mkdir(exist_ok=True)
    
    # Configurar device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {device}")
    
    # Cargar dataset
    dataset = Load_Dataset(df, path, batch_size=16, num_workers=4)
    train_dataloader = dataset.load_train()
    val_dataloader = dataset.load_val()
    test_dataloader = dataset.load_test()
    
    print(f"📦 DataLoaders creados:")
    print(f"   - Train: {len(train_dataloader)} batches")
    print(f"   - Val: {len(val_dataloader)} batches") 
    print(f"   - Test: {len(test_dataloader)} batches")
    
    # ======================== MODELOS A ENTRENAR ========================
    models_to_train = [
        ("MobilnetSegRot", MobilnetSegRot(num_classes=num_classes)),
        ("ResNet50SegRot", ResNet50SegRot(num_classes=num_classes)),
        ("EfficientNetSegRot", EfficientNetSegRot(num_classes=num_classes))
    ]
    
    results = {}
    
    # ======================== ENTRENAMIENTO SECUENCIAL ========================
    log_dirs = {}  # Para guardar los directorios de logs de cada modelo
    
    for model_name, model in models_to_train:
        try:
            print(f"\n🔄 Iniciando entrenamiento de {model_name}...")
            best_f1 = train_model(
                model=model,
                model_name=model_name,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                device=device,
                num_classes=num_classes
            )
            results[model_name] = best_f1
            
            # Guardar el directorio de logs para el resumen final
            model_log_dir = base_log_dir / f'{model_name}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
            log_dirs[model_name] = model_log_dir
            
            # Limpiar memoria VRAM exhaustivamente
            del model
            clear_gpu_memory()
            print(f"🧹 Memoria VRAM liberada después de {model_name}")
            
        except Exception as e:
            print(f"❌ Error entrenando {model_name}: {str(e)}")
            results[model_name] = 0.0
            continue
    
    # ======================== RESUMEN FINAL ========================
    print(f"\n{'='*60}")
    print("📊 RESUMEN DE RESULTADOS")
    print(f"{'='*60}")
    
    for model_name, best_f1 in results.items():
        status = "✅" if best_f1 > 0 else "❌"
        print(f"{status} {model_name}: F1 = {best_f1:.4f}")
    
    # Encontrar el mejor modelo
    if results:
        best_model = max(results, key=results.get)
        print(f"\n🏆 MEJOR MODELO: {best_model} (F1: {results[best_model]:.4f})")
    
    # Mostrar todos los directorios de logs
    print(f"\n� DIRECTORIOS DE LOGS INDIVIDUALES:")
    print(f"{'='*60}")
    for model_name in results.keys():
        # Buscar el directorio más reciente para cada modelo
        model_dirs = list(base_log_dir.glob(f"{model_name}_*"))
        if model_dirs:
            latest_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
            print(f"📊 {model_name}: {latest_dir}")
    
    print(f"\n💡 Para ver los logs en TensorBoard:")
    print(f"   tensorboard --logdir {base_log_dir}")
    print("✅ Entrenamiento secuencial completado!")


if __name__ == "__main__":
    main()