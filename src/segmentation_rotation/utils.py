import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
import random
import numpy as np
import os

TRANSFORM_AUMTENTATION = v2.Compose(
    [
        v2.RandomGrayscale(p=0.5),
        v2.RandomEqualize(p=0.4),
        v2.RandomApply(
            [
                
                #v2.ColorJitter(brightness=0.2, contrast=0.05, saturation=0.1, hue=0.01),
                v2.GaussianBlur(kernel_size=3),
            ],
            p=0.5,
        ),
    ]
)

DATASET_PATH = "/root/Trabajo/Cedulas/imagenes_cedula_peq"

DEFAULT_TRANSFORM = v2.Compose(
    [
        v2.Resize((224, 224), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class Proces_Data(Dataset):
    def __init__(
        self,
        df,
        transform=DEFAULT_TRANSFORM,
        path=DATASET_PATH,
        aumentation=False,
    ) -> None:

        self.df_class = df
        self.path = path
        self.transform = transform
        self.aument = aumentation

    def __len__(self):
        return len(self.df_class)

    def __getitem__(self, idx):
        img_string = f"{self.path}/{self.df_class['Label'].iloc[idx]}/{self.df_class['File'].iloc[idx]}"
        mask_string = f"{self.path}/{self.df_class['Label'].iloc[idx]}_mask/{self.df_class['File'].iloc[idx]}"
        clase = self.df_class["Clase"].iloc[idx]
        angle = self.df_class["Angle"].iloc[idx]
        variation = self.df_class["Variation"].iloc[idx]
        img = Image.open(img_string).convert("RGB")
        mask = Image.open(mask_string).convert("RGB")
        if variation != 0:
            img = img.rotate(angle)
            img = img.rotate(variation)
            mask = mask.rotate(angle)
            mask = mask.rotate(variation)

        img.load()
        mask.load()
        if self.aument:
            img = TRANSFORM_AUMTENTATION(img)
        mask = v2.Compose(
            [
                v2.Grayscale(num_output_channels=1),
                v2.Resize((224, 224), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )(mask)
        img = self.transform(img)

        return img, mask, clase


class Proces_Data_2(Dataset):
    def __init__(
        self,
        df,
        transform=DEFAULT_TRANSFORM,
        path=DATASET_PATH,
        aumentation=False,
    ) -> None:

        self.df_class = df
        self.path = path
        self.transform = transform
        self.aument = aumentation

    def __len__(self):
        return len(self.df_class)

    def __getitem__(self, idx):
        img_string =  img_string = f"{self.path}/Original/{os.path.splitext(self.df_class['File'].iloc[idx])[0]}.jpg"
        mask_string = f"{self.path}/Mascara/{self.df_class['File'].iloc[idx]}"
        clase = self.df_class["Clase"].iloc[idx]
        img = Image.open(img_string).convert("RGB")
        mask = Image.open(mask_string).convert("RGB")
        img.load()
        mask.load()
        random_angle = random.randint(-18000, 18000) / 100.0
        if self.aument:
            img = img.rotate(random_angle)
            mask = mask.rotate(random_angle)
            img = TRANSFORM_AUMTENTATION(img)
        mask = v2.Compose(
            [
                v2.Grayscale(num_output_channels=1),
                v2.Resize((224, 224), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )(mask)
        img = self.transform(img)

        return img, mask, clase


class Load_Dataset:
    def __init__(
        self,
        df: pd.DataFrame = None,
        default_path: str = None,
        val_size: float = 0.25,
        test_size: float = 0.15,
        random_state: int = 42,
        num_workers: int = 2,
        batch_size: int = 8,
    ):
        #### inicio de funcion ####
        df["Unique_ID"], self.labels = pd.factorize(df["Label"])
        self.num_classes = len(self.labels)
        df_train, df_test = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["Label"],
        )
        df_train, df_val = train_test_split(
            df_train,
            test_size=val_size,
            random_state=random_state,
            stratify=df_train["Label"],
        )
        self.cant = []
        for label in self.labels:
            self.cant.append(df_train[df_train["Label"] == label].shape[0])
        self.max_cant = max(self.cant)
        self.path = default_path
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.num_workers = num_workers
        self.batch_size = batch_size

    def load_train(self, transform=DEFAULT_TRANSFORM):
        data_train = Proces_Data(self.df_train, transform, self.path, aumentation=True)
        return DataLoader(
            data_train,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
        )

    def load_val(self, transform=DEFAULT_TRANSFORM):
        data_val = Proces_Data(self.df_val, transform, self.path)
        return DataLoader(
            data_val,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
        )

    def load_test(self, transform=DEFAULT_TRANSFORM):
        data_test = Proces_Data(self.df_test, transform, self.path)
        return DataLoader(
            data_test,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
        )
