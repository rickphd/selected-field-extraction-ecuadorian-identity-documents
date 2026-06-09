import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    vit_b_16,
    ViT_B_16_Weights,
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
)


# Modelo completo Transformer OCR
class OCRModel_Transformer(nn.Module):
    def __init__(self, vocab_size=64, max_len=25, hidden_dim=512, num_layers=2, num_heads=8, sos_idx=1):
        super().__init__()
        self.max_len = max_len
        self.sos_idx = sos_idx
        
        # ---- Encoder: ViT ----
        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.patch_embed = vit.conv_proj
        self.pos_embed = vit.encoder.pos_embedding
        self.transformer_blocks = vit.encoder.layers
        self.norm = vit.encoder.ln
        self.cls_token = nn.Parameter(torch.zeros(1, 1, vit.hidden_dim))
        self.encoder_proj = nn.Linear(vit.hidden_dim, hidden_dim)
        
        # ---- Decoder: Transformer ----
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_decoder = nn.Embedding(max_len, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def encode(self, img):
        B = img.size(0)
        x = self.patch_embed(img)  # (B, C, H/16, W/16)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.pos_embed[:, :x.size(1), :]
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x)
        return self.encoder_proj(x).permute(1, 0, 2)  # (S, B, D)

    def forward(self, img):
        memory = self.encode(img)
        
        B = img.size(0)
        tokens = torch.full((B, self.max_len), self.sos_idx, dtype=torch.long, device=img.device)
        
        # Embedding de salida
        pos = torch.arange(self.max_len, device=img.device).unsqueeze(0)
        tgt_embed = self.embedding(tokens) + self.pos_decoder(pos)
        tgt_embed = tgt_embed.permute(1, 0, 2)  # (T, B, D)
        
        # Decoder
        out = self.decoder(tgt_embed, memory)
        logits = self.output(out)  # (T, B, vocab_size)
        return logits.permute(1, 0, 2)  # (B, T, vocab_size)


# Modelo combinado MobileNetV3 + Segmentación y Clasificación

class MobilnetSeg(nn.Module):
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

