from model import MobilnetSegRot
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



def calculate_iou(predicted_mask, target_mask, epsilon=1e-6):
    predicted_mask = torch.sigmoid(predicted_mask) # probabilidades
    predicted_mask = (predicted_mask > 0.5).float() # matriz true/false -> 1/0
    predicted_mask = predicted_mask.view(-1).bool() # permite hacer operacion & y |
    target_mask = target_mask.view(-1).bool()

    intersection = (predicted_mask & target_mask).float().sum()
    union = (predicted_mask | target_mask).float().sum()

    iou = (intersection + epsilon) / (union + epsilon)
    return iou.item()


def save_model(model, name):
    """Guarda el estado del modelo"""
    torch.save(model.state_dict(), name)


# In[ ]:

if __name__ == "__main__":
    # ======================== CONFIGURACIÓN ========================
    torch.manual_seed(17)
    df_path = Path("../Data_df/progreso_orientacion_aument.csv").resolve()
    path = Path("../Data_img/imagenes_cedula_peq").resolve()
    df = pd.read_csv(df_path).drop(columns=["Unnamed: 0"])
    uniques = np.unique(df['Clase'].values)
    log_dir = Path("runs").resolve() / f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    writer = tb.SummaryWriter(log_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion_mask = nn.BCEWithLogitsLoss()
    criterion_rot = nn.CrossEntropyLoss()
    f1 = F1Score(task="multiclass", num_classes=len(uniques),average="macro").to(device)
    model = MobilnetSegRot(num_classes=len(uniques)).to(device)
    dataset = Load_Dataset(df, path, batch_size=16 , num_workers=4)

    train_dataloader = dataset.load_train()
    val_dataloader = dataset.load_val()
    test_dataloader = dataset.load_test()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
        )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=3.0,
        factor=0.1,
    )


    # In[ ]:


    import os
    # ======================== ENTRENAMIENTO ========================
    best_metric = 0
    patience = 5
    patience_counter = 0
    num_epochs = 200
    add_graf = True
    if os.path.isfile("Mobilnet_mask_rot.pth"):
        model.load_state_dict(torch.load("Mobilnet_mask_rot.pth", map_location=device))
        print("✅ Modelo cargado.")
    else:    
        print("❌ No se encontró el modelo, se entrenará desde cero.")
    model.to(device)    
    for epoch in range(1,num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # ======= TRAIN =======
        model.train()
        running_loss = 0
        train_iou_list = []
        f1_train = 0

        for img, mask , clase in tqdm(train_dataloader, desc="Training", colour="green"):
            img = img.to(device).float()
            mask = mask.to(device).float()
            clase = clase.to(device).long()

            pred_mask , pred_clas = model(img)

            iou = calculate_iou(pred_mask, mask)
            loss_mask = criterion_mask(pred_mask, mask)
            loss_rot = criterion_rot(pred_clas, clase)
            train_iou_list.append(iou)
            f1_train += f1(pred_clas, clase).item()
            total_loss = loss_mask + loss_rot

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

        train_loss = running_loss / len(train_dataloader)
        f1_train /= len(train_dataloader)
        train_iou = np.mean(train_iou_list)

        if writer:
            writer.add_scalar("train_f1", f1_train, epoch)
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("train_mean_iou", train_iou, epoch)
            writer.add_images("train/image", img[:6].cpu(), epoch)
            writer.add_images("train/label", mask[:6].cpu(), epoch)
            writer.add_images("train/pred", torch.sigmoid(pred_mask[:6]).cpu(), epoch)
        print(f"[Train] Loss: {train_loss:.4f} | IoU: {train_iou*100:.2f}% | F1: {f1_train* 100:.2f}%")

        # ======= VALIDATION =======
        model.eval()
        val_loss = 0
        f1_val = 0
        val_iou_list = []

        with torch.no_grad():
            for img, mask , clase in tqdm(val_dataloader, desc="Validation"):
                img = img.to(device).float()
                mask = mask.to(device).float()
                clase = clase.to(device).long()

                pred_mask , pred_clas = model(img)

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
        scheduler.step(val_loss)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
        if writer:
            writer.add_scalar("val_loss", val_loss, epoch)
            writer.add_scalar("val_mean_iou", val_iou, epoch)
            writer.add_scalar("val_f1", f1_val, epoch)
            writer.add_images("val/image", img[:6].cpu(), epoch)
            writer.add_images("val/label", mask[:6].cpu(), epoch)
            writer.add_images("val/pred", torch.sigmoid(pred_mask[:6]).cpu(), epoch)
        print(f"[Val] Loss: {val_loss:.4f} | IoU: {val_iou*100:.2f}% | F1: {f1_val*100:.2f}%")

        # Scheduler y Early Stopping

        if add_graf:
            writer.add_graph(model, img)
            add_graf = False
        if f1_val > best_metric:
            best_metric = f1_val
            patience_counter = 0
            save_model(model=model, name="Mobilnet_mask_rot.pth")
            print("✅ Modelo guardado.")
        else:
            if optimizer.param_groups[0]["lr"] < 2e-8:
                patience_counter += 1
                print(f"🔻 Reducción de LR {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("🛑 Early stopping")
                break


