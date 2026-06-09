import torch, os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import v2
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import random

size = (224 // 2, 224)  # Height, Width

AUMENTATION_TRANSFORM = v2.Compose(
    [
        v2.RandomInvert(p=0.5),
        v2.RandomRotation(degrees=3, expand=False, fill=0),
        v2.RandomAffine(degrees=0, scale=(0.95, 1.05), shear=5, fill=0),
        #v2.RandomPerspective(distortion_scale=0.2, p=0.5, fill=0),
        v2.Resize(size=size),
        v2.RandomApply(
            [
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                v2.GaussianBlur(kernel_size=3),
            ],
            p=0.5,
        ),
        v2.RandomHorizontalFlip(p=0.1),
        v2.RandomVerticalFlip(p=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0, 0, 0], 
            std=[1, 1, 1],
        ),
    ]
)

default_transform = v2.Compose(
    [
        v2.Resize(size=size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0, 0, 0], 
            std=[1, 1, 1]
        ),
    ]
)


class ImageDataset(Dataset):
    def __init__(
        self,
        df,
        transform=default_transform,
        base_path=None,
        augmentation=False,
    ):
        self.df = df
        self.transform = transform
        self.base_path = base_path
        self.augmentation = augmentation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.df["FILE"].iloc[idx]
        label = self.df["TEXT"].iloc[idx]
        img_path = os.path.join(self.base_path, f"{image}")
        # print(img_path)
        image = Image.open(img_path).convert("RGB")
        if self.augmentation:
            if random.random() < 0.4:  # 40% de probabilidad
                image = image.rotate(180)  # Rotación aleatoria entre 180
            image = AUMENTATION_TRANSFORM(image)
        else:
            image = self.transform(image)
        return image, label.replace("-", "").upper()


class ImageDataset_2(Dataset):
    def __init__(self, df, transform=default_transform, augmentation=False):
        self.df = df
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.df["NOMBRE_IMAGEN"][idx]
        label = self.df["TEXTO"][idx]
        label = label.upper()
        if self.augmentation:
            if random.random() < 0.3:  # 30% de probabilidad
                image = AUMENTATION_TRANSFORM(image.rotate(180))
            else:
                image = AUMENTATION_TRANSFORM(image)
        else:
            image = self.transform(image)
        return image, label


class ImageDataLoader(DataLoader):
    def __init__(
        self,
        df,
        batch_size=32,
        num_workers=2,
        base_path=None,
    ):
        self.df_train, self.df_val = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.base_path = base_path

    def data_train(self):
        train_dataset = ImageDataset(
            self.df_train,
            base_path=self.base_path,
            augmentation=True,
        )
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

    def data_val(self):
        val_dataset = ImageDataset(
            self.df_val,
            base_path=self.base_path,
        )
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
        )


class ImageDataLoader_2(DataLoader):
    def __init__(
        self,
        df,
        batch_size=32,
        num_workers=2,
    ):
        self.df_train, self.df_val = df["train"], df["test"]
        self.batch_size = batch_size
        self.num_workers = num_workers

    def data_train(self):
        train_dataset = ImageDataset_2(
            self.df_train,
            augmentation=True,
        )
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

    def data_val(self):
        val_dataset = ImageDataset_2(
            self.df_val,
        )
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
        )
