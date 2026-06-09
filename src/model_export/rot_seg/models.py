import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    resnet50, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
) 

class MobilnetSegRot(nn.Module):
    def __init__(self, num_classes=1 , freeze_Enc=False , freeze_classif=False , freeze_seg=False):
        super().__init__()

        # Cargar MobileNet-V3 preentrenada
        mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        self.transform = MobileNet_V3_Large_Weights.IMAGENET1K_V2.transforms()
        # Encoder: solo las features
        self.encoder = mobilenet.features # (B, 1280, H/32, W/32)
        self.latent_channels = self.encoder[-1].out_channels
        # Cabeza de rotacion  
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_channels, num_classes),
            nn.LayerNorm(num_classes),
            nn.Dropout(0.2)
        )

        # Decoder para segmentación (upsample 5 veces para volver a HxW)
        self.seg_decoder = nn.Sequential(
            nn.ConvTranspose2d(self.latent_channels, 512, 4, 2, 1),  # H/16
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),   # H/8
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),   # H/4
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),    # H/2
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),      # H
        )
        self.final_conv = nn.Sequential(
                nn.Conv2d(32, 32//2, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(32//2, 1, kernel_size=3, padding=1),
                nn.LeakyReLU(),
            )
        if freeze_Enc:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if freeze_classif:
            for param in self.classifier.parameters():
                param.requires_grad = False
        if freeze_seg:
            for param in self.seg_decoder.parameters():
                param.requires_grad = False
            for param in self.final_conv.parameters():
                param.requires_grad = False
        

    def forward(self, x):
        feats = self.encoder(x)          # (B, 1280, H/32, W/32)
        avg_pool = self.avgpool(feats)  # (B, 1280, 1, 1)
        logits = torch.flatten(avg_pool, 1)  # (B, 1280)
        class_out = self.classifier(logits)   # (B, num_classes)
        seg_out = self.seg_decoder(feats)
        seg_out = self.final_conv(seg_out)  # (B, 1, H, W)
        seg_out = F.interpolate(seg_out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return seg_out , class_out


class ResNet50SegRot(nn.Module):
    def __init__(self, num_classes=1, freeze_Enc=False, freeze_classif=False, freeze_seg=False):
        super().__init__()

        # Cargar ResNet-50 preentrenada
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
        
        # Encoder: todas las capas menos el clasificador final
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.latent_channels = 2048  # ResNet-50 output channels
        
        # Cabeza de rotación/clasificación  
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_channels, num_classes),
            nn.LayerNorm(num_classes),
            nn.Dropout(0.2)
        )

        # Decoder para segmentación (upsample 5 veces para volver a HxW)
        self.seg_decoder = nn.Sequential(
            nn.ConvTranspose2d(self.latent_channels, 1024, 4, 2, 1),  # H/16
            nn.LeakyReLU(),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),   # H/8
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),   # H/4
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),    # H/2
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),      # H
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )
        
        if freeze_Enc:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if freeze_classif:
            for param in self.classifier.parameters():
                param.requires_grad = False
        if freeze_seg:
            for param in self.seg_decoder.parameters():
                param.requires_grad = False
            for param in self.final_conv.parameters():
                param.requires_grad = False

    def forward(self, x):
        feats = self.encoder(x)          # (B, 2048, H/32, W/32)
        avg_pool = self.avgpool(feats)   # (B, 2048, 1, 1)
        logits = torch.flatten(avg_pool, 1)  # (B, 2048)
        class_out = self.classifier(logits)   # (B, num_classes)
        seg_out = self.seg_decoder(feats)
        seg_out = self.final_conv(seg_out)  # (B, 1, H, W)
        seg_out = F.interpolate(seg_out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return seg_out, class_out


class EfficientNetSegRot(nn.Module):
    def __init__(self, num_classes=1, freeze_Enc=False, freeze_classif=False, freeze_seg=False):
        super().__init__()

        # Cargar EfficientNet-B0 preentrenada
        efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.transform = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
        
        # Encoder: solo las features
        self.encoder = efficientnet.features  # (B, 1280, H/32, W/32)
        self.latent_channels = 1280  # EfficientNet-B0 output channels
        
        # Cabeza de rotación/clasificación  
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_channels, num_classes),
            nn.LayerNorm(num_classes),
            nn.Dropout(0.2)
        )

        # Decoder para segmentación (upsample 5 veces para volver a HxW)
        self.seg_decoder = nn.Sequential(
            nn.ConvTranspose2d(self.latent_channels, 640, 4, 2, 1),  # H/16
            nn.LeakyReLU(),
            nn.ConvTranspose2d(640, 320, 4, 2, 1),   # H/8
            nn.LeakyReLU(),
            nn.ConvTranspose2d(320, 160, 4, 2, 1),   # H/4
            nn.LeakyReLU(),
            nn.ConvTranspose2d(160, 80, 4, 2, 1),    # H/2
            nn.LeakyReLU(),
            nn.ConvTranspose2d(80, 40, 4, 2, 1),      # H
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(40, 20, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(20, 1, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )
        
        if freeze_Enc:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if freeze_classif:
            for param in self.classifier.parameters():
                param.requires_grad = False
        if freeze_seg:
            for param in self.seg_decoder.parameters():
                param.requires_grad = False
            for param in self.final_conv.parameters():
                param.requires_grad = False

    def forward(self, x):
        feats = self.encoder(x)          # (B, 1280, H/32, W/32)
        avg_pool = self.avgpool(feats)   # (B, 1280, 1, 1)
        logits = torch.flatten(avg_pool, 1)  # (B, 1280)
        class_out = self.classifier(logits)   # (B, num_classes)
        seg_out = self.seg_decoder(feats)
        seg_out = self.final_conv(seg_out)  # (B, 1, H, W)
        seg_out = F.interpolate(seg_out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return seg_out, class_out