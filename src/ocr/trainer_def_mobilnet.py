#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 Entrenamiento Secuencial de Modelos OCR
==========================================
Entrenador optimizado para entrenar todos los modelos OCR secuencialmente
con gestión inteligente de memoria VRAM y early stopping avanzado.
"""

# 📦 Importaciones necesarias
from models import (
    OCRModel_LSTM_CustomBackbone,
    OCRModel_GRU_CustomBackbone,
    OCRModel_LSTM_MobileNet,
    OCRModel_GRU_MobileNet,
    OCRModel_Transformer
)
from utils import ImageDataLoader
from text_converter import TextConverter
import torch
import torch.nn as nn
import torchmetrics
import pandas as pd
from tqdm import tqdm
import os
import gc
import shutil
from torch.utils.tensorboard import SummaryWriter

# � Configuración Global
# =======================

TRAINING_CONFIG = {
    'batch_size': 16,
    'num_workers': 8,
    'max_len': 25,
    'epochs': 300,  # Épocas aumentadas para entrenamiento real
    'patience': 5,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'label_smoothing': 0.1,
    'min_lr': 1.5e-8  # Learning rate mínimo para early stopping
}

# Modelos disponibles
AVAILABLE_MODELS = {
    #'lstm_custom': OCRModel_LSTM_CustomBackbone,
    #'gru_custom': OCRModel_GRU_CustomBackbone,
    'lstm_mobilenet': OCRModel_LSTM_MobileNet,
    'gru_mobilenet': OCRModel_GRU_MobileNet,
    #'transformer': OCRModel_Transformer,
}

# 📊 Funciones Auxiliares
# =======================

def unir_csv():
    """Función para unir todos los datasets CSV en uno solo"""
    print("📁 Cargando datasets...")
    
    # Cargar todos los datasets
    df_all = pd.read_csv("Data_df/labels_all.csv")
    df_excelent = pd.read_csv("Data_df/results_new.csv")
    df_0 = pd.read_csv("Data_df/resultados_recorte_clean_0.csv")
    df_m_0 = pd.read_csv("Data_df/resultados_recorte_clean_M_0.csv")
    df_m_1 = pd.read_csv("Data_df/resultados_recorte_clean_M_1.csv")
    
    print(f"  - labels_all.csv: {len(df_all)} muestras")
    print(f"  - results_new.csv: {len(df_excelent)} muestras")
    print(f"  - recorte_clean_0.csv: {len(df_0)} muestras")
    print(f"  - recorte_clean_M_0.csv: {len(df_m_0)} muestras")
    print(f"  - recorte_clean_M_1.csv: {len(df_m_1)} muestras")
    
    # Convertir formatos para que sean consistentes
    df_all_convert = pd.DataFrame({
        "FILE": df_all["image"].apply(lambda x: f"all_images/{x}"),
        "TEXT": df_all["text"],
    })
    
    df_excelent_convert = pd.DataFrame({
        "FILE": df_excelent["image_path"].apply(lambda x: f"recortes_txt/{x}"),
        "TEXT": df_excelent["text"],
    })
    
    # Unir todos los datasets
    df_final = pd.concat([
        df_0, df_m_0, df_m_1, 
        df_all_convert, df_excelent_convert
    ], ignore_index=True)
    
    print(f"📊 Dataset final: {len(df_final)} muestras totales")
    print(f"📏 Longitud promedio del texto: {df_final['TEXT'].str.len().mean():.1f} caracteres")
    
    return df_final

def create_model(model_name, converter, max_len, device):
    """Crea e inicializa un modelo específico"""
    print(f"🏗️ Inicializando modelo: {model_name}")
    
    if model_name == 'transformer':
        model = AVAILABLE_MODELS[model_name](
            vocab_size=converter.vocab_size(),
            max_len=max_len,
            hidden_dim=512,
            num_layers=2,
            num_heads=8,
            sos_idx=converter.char2idx.get('<SOS>', 1)
        )
    else:
        model = AVAILABLE_MODELS[model_name](
            vocab_size=converter.vocab_size(),
            hidden_size=512,
            num_layers=2,
            max_len=max_len
        )
    
    model = model.to(device)
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  📊 Parámetros totales: {total_params:,}")
    print(f"  📊 Parámetros entrenables: {trainable_params:,}")
    print(f"  📊 Tamaño del modelo: {total_params * 4 / (1024**2):.1f} MB")
    
    return model, total_params

def train_single_model(model_name, model, train_loader, val_loader, converter, device):
    """Entrena un modelo individual y retorna los resultados"""
    
    print(f"\n{'='*80}")
    print(f"🚀 INICIANDO ENTRENAMIENTO: {model_name.upper()}")
    print(f"{'='*80}")
    
    # Métricas
    accuracy = torchmetrics.Accuracy(
        task="multiclass", 
        num_classes=converter.vocab_size()
    ).to(device)
    
    cer = torchmetrics.CharErrorRate().to(device)
    
    # Función de pérdida
    criterion = nn.CrossEntropyLoss(
        ignore_index=converter.char2idx["<PAD>"],
        label_smoothing=TRAINING_CONFIG['label_smoothing']
    )
    
    # Optimizador y scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG['lr'],
        weight_decay=TRAINING_CONFIG['weight_decay'],
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=3,
    )
    
    # Configuración de logs - UNA CARPETA POR MODELO
    logs_base = "./runs"
    os.makedirs(logs_base, exist_ok=True)
    
    # Directorio simple por modelo (sin numeración)
    log_dir = os.path.join(logs_base, model_name)
    
    # Limpiar carpeta anterior si existe
    if os.path.exists(log_dir):
        import shutil
        shutil.rmtree(log_dir)
        print(f"🧹 Limpiando logs anteriores de {model_name}")
    
    writer = SummaryWriter(log_dir)
    print(f"📊 TensorBoard logs: {log_dir}")
    
    # Variables para early stopping
    best_cer = float('inf')
    patience_counter = 0
    model_save_path = f"best_model_{model_name}.pth"
    
    # Cargar modelo previo si existe
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"✅ Modelo cargado desde {model_save_path}")
    else:
        print(f"🆕 Entrenando modelo desde cero")
    
    # Listas para historial
    train_losses = []
    val_losses = []
    train_cers = []
    val_cers = []
    
    return {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'criterion': criterion,
        'accuracy': accuracy,
        'cer': cer,
        'writer': writer,
        'best_cer': best_cer,
        'patience_counter': patience_counter,
        'model_save_path': model_save_path,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_cers': train_cers,
        'val_cers': val_cers
    }

print("🎯 Función de entrenamiento individual configurada")

# %%
# 🏃‍♂️ Entrenamiento Secuencial de Todos los Modelos
# =================================================

def run_epoch(model, train_loader, val_loader, training_components, epoch, converter, device):
    """Ejecuta una época completa de entrenamiento y validación"""
    
    model = training_components['model']
    optimizer = training_components['optimizer']
    criterion = training_components['criterion']
    accuracy = training_components['accuracy']
    cer = training_components['cer']
    
    # ========================
    # 🚂 FASE DE ENTRENAMIENTO
    # ========================
    model.train()
    train_loss = 0.0
    train_cer_score = 0.0
    train_acc_score = 0.0
    
    # Variable para guardar imágenes del primer batch
    first_batch_images = None
    first_batch_labels = None
    
    train_pbar = tqdm(train_loader, desc=f"Época {epoch:2d}/{TRAINING_CONFIG['epochs']} - Entrenamiento", leave=False)
    
    for batch_idx, (images, labels) in enumerate(train_pbar):
        try:
            images = images.to(device, non_blocking=True)
            
            # Capturar las primeras 6 imágenes del primer batch para logging
            if batch_idx == 0 and first_batch_images is None:
                first_batch_images = images[:6].clone().cpu()  # Solo las primeras 6
                first_batch_labels = labels[:6]  # Las etiquetas correspondientes
            
            # Verificar que las imágenes no tengan valores problemáticos
            if torch.isnan(images).any() or torch.isinf(images).any():
                print(f"⚠️ Batch {batch_idx}: Valores problemáticos en imágenes, saltando...")
                continue
            
            # Codificar las etiquetas
            encoded = [converter.encode(text, max_len=model.max_len) for text in labels]
            targets = torch.tensor(encoded, dtype=torch.long, device=device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Verificar outputs
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f"⚠️ Batch {batch_idx}: Outputs problemáticos, saltando...")
                continue
            
            # Calcular pérdida
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            
            # Verificar pérdida
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ Batch {batch_idx}: Pérdida problemática ({loss.item()}), saltando...")
                continue
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Calcular métricas
            with torch.no_grad():
                pred_tokens = outputs.argmax(-1)
                decoded_preds = [converter.decode(tokens) for tokens in pred_tokens.cpu().numpy()]
                
                train_loss += loss.item()
                train_acc_score += accuracy(pred_tokens, targets).item()
                train_cer_score += cer(decoded_preds, labels).item()
            
            # Actualizar barra
            if batch_idx % 10 == 0:
                current_loss = train_loss / (batch_idx + 1)
                current_cer = train_cer_score / (batch_idx + 1) * 100
                train_pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'CER': f'{current_cer:.2f}%'
                })
                
        except RuntimeError as e:
            if "lstsq" in str(e) or "perspective" in str(e) or "full rank" in str(e):
                print(f"⚠️ Error de transformación en batch {batch_idx}, saltando...")
                continue
            else:
                print(f"❌ Error crítico en batch {batch_idx}: {e}")
                raise e
        except Exception as e:
            print(f"⚠️ Error inesperado en batch {batch_idx}: {e}, saltando...")
            continue
    
    # ========================
    # ✅ FASE DE VALIDACIÓN
    # ========================
    model.eval()
    val_loss = 0.0
    val_cer_score = 0.0
    val_acc_score = 0.0
    
    val_pbar = tqdm(val_loader, desc=f"Época {epoch:2d}/{TRAINING_CONFIG['epochs']} - Validación", leave=False)
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_pbar):
            try:
                images = images.to(device, non_blocking=True)
                
                # Verificar que las imágenes sean válidas
                if torch.isnan(images).any() or torch.isinf(images).any():
                    print(f"⚠️ Val Batch {batch_idx}: Valores problemáticos, saltando...")
                    continue
                
                encoded = [converter.encode(text, max_len=model.max_len) for text in labels]
                targets = torch.tensor(encoded, dtype=torch.long, device=device)
                
                outputs = model(images)
                
                # Verificar outputs
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"⚠️ Val Batch {batch_idx}: Outputs problemáticos, saltando...")
                    continue
                
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
                
                # Verificar pérdida
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ Val Batch {batch_idx}: Pérdida problemática, saltando...")
                    continue
                
                pred_tokens = outputs.argmax(-1)
                decoded_preds = [converter.decode(tokens) for tokens in pred_tokens.cpu().numpy()]
                
                val_loss += loss.item()
                val_acc_score += accuracy(pred_tokens, targets).item()
                val_cer_score += cer(decoded_preds, labels).item()
                
                if batch_idx % 10 == 0:
                    current_loss = val_loss / (batch_idx + 1)
                    current_cer = val_cer_score / (batch_idx + 1) * 100
                    val_pbar.set_postfix({
                        'Loss': f'{current_loss:.4f}',
                        'CER': f'{current_cer:.2f}%'
                    })
                    
            except RuntimeError as e:
                if "lstsq" in str(e) or "perspective" in str(e) or "full rank" in str(e):
                    print(f"⚠️ Error de transformación en val batch {batch_idx}, saltando...")
                    continue
                else:
                    print(f"❌ Error crítico en val batch {batch_idx}: {e}")
                    raise e
            except Exception as e:
                print(f"⚠️ Error inesperado en val batch {batch_idx}: {e}, saltando...")
                continue
    
    # Calcular promedios
    avg_train_loss = train_loss / len(train_loader)
    avg_train_cer = train_cer_score / len(train_loader) * 100
    avg_train_acc = train_acc_score / len(train_loader) * 100
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_cer = val_cer_score / len(val_loader) * 100
    avg_val_acc = val_acc_score / len(val_loader) * 100
    
    return {
        'train_loss': avg_train_loss,
        'train_cer': avg_train_cer,
        'train_acc': avg_train_acc,
        'val_loss': avg_val_loss,
        'val_cer': avg_val_cer,
        'val_acc': avg_val_acc,
        'decoded_preds': decoded_preds,
        'labels': labels,
        'first_batch_images': first_batch_images,
        'first_batch_labels': first_batch_labels
    }


def main():
    """Función principal del entrenamiento secuencial"""
    
    # 🔥 Configuración del dispositivo
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("✅ Librerías importadas correctamente")
    print(f"🔥 CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    print(f"🔥 Dispositivo seleccionado: {device}")
    
    print("📋 Configuración de entrenamiento:")
    for key, value in TRAINING_CONFIG.items():
        print(f"  - {key}: {value}")
    
    # 📊 Preparar datos
    df = unir_csv()
    print("\n✅ Datos preparados correctamente")
    
    # 📝 Inicializar convertidor de texto
    converter = TextConverter()
    print(f"📝 Vocabulario: {converter.vocab_size()} caracteres")
    
    print("\n🧠 Modelos disponibles para entrenamiento secuencial:")
    for model_name in AVAILABLE_MODELS.keys():
        print(f"  - {model_name}")
    
    # Variables globales para tracking
    training_results = {}
    overall_best_model = None
    overall_best_cer = float('inf')
    
    # Llamar a la función de entrenamiento con todas las variables necesarias
    train_all_models_sequential(
        df, converter, device, training_results, 
        overall_best_model, overall_best_cer
    )


def train_all_models_sequential(df, converter, device, training_results, overall_best_model, overall_best_cer):
    """Función principal de entrenamiento secuencial de todos los modelos"""
    
    # 🔧 Preparación Segura del DataLoader (código movido aquí)
    base_path = "./"
    
    # Configuración segura para evitar errores de transformación
    SAFE_LOADER_CONFIG = {
        'batch_size': TRAINING_CONFIG['batch_size'],
        'num_workers': min(TRAINING_CONFIG['num_workers'], 2),
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': True if TRAINING_CONFIG['num_workers'] > 0 else False,
        'prefetch_factor': 2 if TRAINING_CONFIG['num_workers'] > 0 else None
    }
    
    print("🔧 Configuración segura del DataLoader:")
    for key, value in SAFE_LOADER_CONFIG.items():
        if value is not None:
            print(f"  - {key}: {value}")

    try:
        # Crear DataLoader con configuración segura
        print("\n📁 Creando DataLoader...")
        loader = ImageDataLoader(
            df, 
            batch_size=SAFE_LOADER_CONFIG['batch_size'],
            num_workers=SAFE_LOADER_CONFIG['num_workers'],
            base_path=base_path
        )
        
        # Obtener los loaders con manejo de errores
        print("🚂 Configurando loaders de entrenamiento y validación...")
        train_loader = loader.data_train()
        val_loader = loader.data_val()
        
        print(f"✅ Datos de entrenamiento: {len(train_loader)} batches")
        print(f"✅ Datos de validación: {len(val_loader)} batches")
        print(f"📊 Tamaño del batch: {SAFE_LOADER_CONFIG['batch_size']}")
        
        # Verificación segura de una muestra
        print("\n🔍 Verificando una muestra de forma segura...")
        sample_success = False
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                for images, labels in train_loader:
                    print(f"📷 Forma de las imágenes: {images.shape}")
                    print(f"📝 Ejemplo de etiquetas: {labels[:3]}")
                    print(f"📐 Entrada esperada: (batch_size, 3, 112, 224)")
                    print(f"✅ Verificación exitosa en intento {attempt + 1}")
                    sample_success = True
                    break
                if sample_success:
                    break
            except Exception as e:
                print(f"⚠️ Intento {attempt + 1} falló: {str(e)[:100]}...")
                if attempt < max_attempts - 1:
                    print("🔄 Reintentando...")
                continue
        
        if not sample_success:
            print("❌ No se pudo verificar muestra. Creando DataLoader alternativo...")
            
            # Configuración alternativa más conservadora
            SAFE_LOADER_CONFIG['num_workers'] = 0
            SAFE_LOADER_CONFIG['persistent_workers'] = False
            SAFE_LOADER_CONFIG['pin_memory'] = False
            
            loader = ImageDataLoader(
                df, 
                batch_size=SAFE_LOADER_CONFIG['batch_size'],
                num_workers=0,
                base_path=base_path
            )
            
            train_loader = loader.data_train()
            val_loader = loader.data_val()
            print("✅ DataLoader alternativo creado exitosamente")

    except Exception as e:
        print(f"❌ Error creando DataLoader: {e}")
        print("🔄 Intentando con configuración mínima...")
        
        # Configuración de emergencia
        try:
            loader = ImageDataLoader(
                df.head(100),  # Usar solo una muestra pequeña
                batch_size=4,   # Batch pequeño
                num_workers=0,  # Sin multiprocessing
                base_path=base_path
            )
            
            train_loader = loader.data_train()
            val_loader = loader.data_val()
            
            print("⚠️ DataLoader de emergencia creado (muestra reducida)")
            print(f"📊 Datos limitados para prueba: {len(train_loader)} batches de entrenamiento")
            
        except Exception as e2:
            print(f"💥 Error crítico: {e2}")
            print("🛑 No se puede continuar con el DataLoader actual")
            raise e2

    print("\n✅ DataLoader configurado exitosamente")
    
    # ========================
    # 🌟 ENTRENAMIENTO SECUENCIAL
    # ========================
    
    print(f"🌟 INICIANDO ENTRENAMIENTO SECUENCIAL DE TODOS LOS MODELOS")
    print(f"📊 Número de modelos a entrenar: {len(AVAILABLE_MODELS)}")
    print(f"🏋️‍♂️ Épocas por modelo: {TRAINING_CONFIG['epochs']}")
    print(f"⏰ Paciencia: {TRAINING_CONFIG['patience']}")
    print(f"🔻 Learning rate mínimo: {TRAINING_CONFIG['min_lr']:.2e}")
    print("=" * 80)

    for model_idx, model_name in enumerate(AVAILABLE_MODELS.keys(), 1):
        
        print(f"\n🧠 MODELO {model_idx}/{len(AVAILABLE_MODELS)}: {model_name.upper()}")
        print("-" * 60)
        
        # ========================
        # 📊 MONITOREO DE MEMORIA INICIAL
        # ========================
        if torch.cuda.is_available():
            # Limpiar memoria antes de empezar
            torch.cuda.empty_cache()
            
            # Información de memoria inicial
            mem_free, mem_total = torch.cuda.mem_get_info()
            mem_used = mem_total - mem_free
            
            print(f"  🎮 GPU: {torch.cuda.get_device_name()}")
            print(f"  📊 Memoria total: {mem_total / (1024**3):.2f} GB")
            print(f"  📊 Memoria usada: {mem_used / (1024**3):.2f} GB")
            print(f"  📊 Memoria libre: {mem_free / (1024**3):.2f} GB")
            print(f"  📊 Uso: {(mem_used / mem_total) * 100:.1f}%")
        
        # Crear modelo
        model, total_params = create_model(model_name, converter, TRAINING_CONFIG['max_len'], device)
        
        # Monitoreo después de cargar modelo
        if torch.cuda.is_available():
            mem_after_model = torch.cuda.memory_allocated() / (1024**3)  # GB
            print(f"  📊 Memoria del modelo: {mem_after_model:.2f} GB")
        
        # Configurar entrenamiento
        training_components = train_single_model(model_name, model, train_loader, val_loader, converter, device)
        
        # Variables de control
        best_cer = training_components['best_cer']
        patience_counter = training_components['patience_counter']
        model_save_path = training_components['model_save_path']
        writer = training_components['writer']
        scheduler = training_components['scheduler']
        
        # Entrenar por épocas
        print(f"🏋️‍♂️ Iniciando entrenamiento por {TRAINING_CONFIG['epochs']} épocas...")
        
        for epoch in range(1, TRAINING_CONFIG['epochs'] + 1):
            
            # Ejecutar época
            epoch_results = run_epoch(model, train_loader, val_loader, training_components, epoch, converter, device)
            
            # Actualizar scheduler
            scheduler.step(epoch_results['val_cer'])
            current_lr = training_components['optimizer'].param_groups[0]['lr']
            
            # Logging de métricas
            writer.add_scalar("Loss/train", epoch_results['train_loss'], epoch)
            writer.add_scalar("Loss/val", epoch_results['val_loss'], epoch)
            writer.add_scalar("CER/train", epoch_results['train_cer'], epoch)
            writer.add_scalar("CER/val", epoch_results['val_cer'], epoch)
            writer.add_scalar("Accuracy/train", epoch_results['train_acc'], epoch)
            writer.add_scalar("Accuracy/val", epoch_results['val_acc'], epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Guardar las 6 primeras imágenes del primer batch (solo una vez)
            if epoch == 1 and epoch_results['first_batch_images'] is not None:
                # Desnormalizar las imágenes para visualización
                images_to_log = epoch_results['first_batch_images'].clone()
                
                # Desnormalizar (revertir normalización de ImageNet)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                images_to_log = images_to_log * std + mean
                images_to_log = torch.clamp(images_to_log, 0, 1)  # Asegurar rango [0,1]
                
                # Agregar a TensorBoard con etiquetas
                writer.add_images(
                    'Training_Samples/First_Batch_Images',
                    images_to_log, 
                    global_step=0
                )
                
                # Agregar texto con las etiquetas correspondientes
                labels_text = "\\n".join([f"Imagen {i+1}: '{label}'" 
                                         for i, label in enumerate(epoch_results['first_batch_labels'])])
                writer.add_text('Training_Samples/First_Batch_Labels', labels_text, global_step=0)
                
                print(f"  📸 Guardadas 6 imágenes de muestra en TensorBoard")
            
            # Imprimir estadísticas
            print(f"📊 Época {epoch:2d}/{TRAINING_CONFIG['epochs']} - {model_name}:")
            print(f"  🚂 Train - Loss: {epoch_results['train_loss']:.4f}, CER: {epoch_results['train_cer']:.2f}%, Acc: {epoch_results['train_acc']:.2f}%")
            print(f"  ✅ Val   - Loss: {epoch_results['val_loss']:.4f}, CER: {epoch_results['val_cer']:.2f}%, Acc: {epoch_results['val_acc']:.2f}%")
            print(f"  📈 LR: {current_lr:.2e}")
            
            # Early stopping y guardado
            if epoch_results['val_cer'] < best_cer:
                best_cer = epoch_results['val_cer']
                patience_counter = 0
                
                # Guardar el mejor modelo
                torch.save(model.state_dict(), model_save_path)
                print(f"  💾 ¡Nuevo mejor modelo guardado! CER: {best_cer:.2f}%")
                
                # Actualizar mejor modelo global
                if best_cer < overall_best_cer:
                    overall_best_cer = best_cer
                    overall_best_model = model_name
                    print(f"  🌟 ¡NUEVO MEJOR MODELO GLOBAL! {model_name}: {best_cer:.2f}%")
                
                # Guardar ejemplos
                writer.add_text('Examples/Predictions', 
                               f'Epoch {epoch}:\\n' + '\\n'.join([f'GT: {gt} | Pred: {pred}' 
                                                                for gt, pred in zip(epoch_results['labels'][:3], 
                                                                                   epoch_results['decoded_preds'][:3])]), 
                               epoch)
            else:
                # Solo contar paciencia si el learning rate ya es muy pequeño
                if current_lr < TRAINING_CONFIG['min_lr']:
                    patience_counter += 1
                    print(f"  ⏰ Sin mejora con LR pequeño ({current_lr:.2e}). Paciencia: {patience_counter}/{TRAINING_CONFIG['patience']}")
                else:
                    print(f"  📊 Sin mejora, pero LR aún alto ({current_lr:.2e}). Paciencia en espera.")
            
            # Early stopping por paciencia (solo cuando LR es pequeño)
            if patience_counter >= TRAINING_CONFIG['patience'] and current_lr < TRAINING_CONFIG['min_lr']:
                print(f"  🛑 Early stopping: {TRAINING_CONFIG['patience']} épocas sin mejora con LR muy pequeño.")
                print(f"  🏆 Mejor CER para {model_name}: {best_cer:.2f}%")
                break
            
            print("-" * 40)
        
        # Guardar resultados del modelo
        training_results[model_name] = {
            'best_cer': best_cer,
            'total_params': total_params,
            'model_path': model_save_path,
            'epochs_trained': epoch
        }
        
        # Cerrar writer
        writer.close()
        
        # ========================
        # 🧹 LIBERACIÓN AGRESIVA DE MEMORIA VRAM
        # ========================
        print(f"\n🧹 Liberando memoria VRAM para {model_name}...")
        
        if torch.cuda.is_available():
            # Memoria antes de limpiar
            mem_before = torch.cuda.memory_allocated() / (1024**3)  # GB
            print(f"  📊 Memoria VRAM antes: {mem_before:.2f} GB")
        
        # 1. Mover modelo a CPU y eliminar
        model = model.cpu()
        del model
        
        # 2. Limpiar componentes de entrenamiento
        del training_components['model']
        del training_components['optimizer'] 
        del training_components['scheduler']
        del training_components['criterion']
        del training_components['accuracy']
        del training_components['cer']
        del training_components
        
        # 3. Forzar garbage collection
        gc.collect()
        
        # 4. Limpiar cache de CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # Limpiar memoria compartida IPC
            
            # Memoria después de limpiar
            mem_after = torch.cuda.memory_allocated() / (1024**3)  # GB
            mem_freed = mem_before - mem_after
            
            print(f"  📊 Memoria VRAM después: {mem_after:.2f} GB")
            print(f"  ♻️ Memoria liberada: {mem_freed:.2f} GB")
            
            # Información adicional de memoria
            mem_reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
            mem_max = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            
            print(f"  📈 Memoria reservada: {mem_reserved:.2f} GB")
            print(f"  📊 Pico máximo usado: {mem_max:.2f} GB")
            
            # Reset estadísticas de memoria
            torch.cuda.reset_peak_memory_stats()
        
        print(f"✅ MODELO {model_name} COMPLETADO - Mejor CER: {best_cer:.2f}%")
        print(f"🧹 Memoria VRAM liberada exitosamente")
        print("=" * 60)

    # ========================
    # 🏆 RESUMEN FINAL
    # ========================

    print(f"\n🎉 ¡ENTRENAMIENTO SECUENCIAL COMPLETADO!")
    print("=" * 80)
    print(f"🌟 MEJOR MODELO GLOBAL: {overall_best_model} (CER: {overall_best_cer:.2f}%)")
    print("=" * 80)

    print("\n📊 RESULTADOS POR MODELO:")
    print("-" * 60)
    for model_name, results in training_results.items():
        print(f"🧠 {model_name:<15}: CER={results['best_cer']:6.2f}% | Params={results['total_params']:>8,} | Épocas={results['epochs_trained']:2d}")

    print("-" * 60)

    # ========================
    # 🧹 LIMPIEZA FINAL DE MEMORIA
    # ========================
    if torch.cuda.is_available():
        print(f"\n🧹 LIMPIEZA FINAL DE MEMORIA VRAM:")
        print("-" * 40)
        
        # Limpiar cualquier resto en memoria
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # Estado final de memoria
        mem_free, mem_total = torch.cuda.mem_get_info()
        mem_used = mem_total - mem_free
        mem_allocated = torch.cuda.memory_allocated() / (1024**3)
        
        print(f"  📊 Memoria total GPU: {mem_total / (1024**3):.2f} GB")
        print(f"  📊 Memoria libre: {mem_free / (1024**3):.2f} GB") 
        print(f"  📊 Memoria usada por sistema: {mem_used / (1024**3):.2f} GB")
        print(f"  📊 Memoria PyTorch: {mem_allocated:.2f} GB")
        print(f"  ♻️ Estado: {'✅ Limpia' if mem_allocated < 0.1 else '⚠️ Residual'}")
        print("-" * 40)

    print(f"\n💾 Todos los modelos guardados en sus respectivos archivos .pth")
    print(f"📊 Logs de TensorBoard disponibles en ./runs/")
    print("✨ ¡Entrenamiento secuencial finalizado exitosamente! ✨")


if __name__ == "__main__":
    main()


