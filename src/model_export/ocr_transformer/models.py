import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
    vit_b_16,
    ViT_B_16_Weights,
)


# Convert maxpool3d to the class of maxpool2d
class maxpool_3d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(maxpool_3d, self).__init__()
        assert(len(kernel_size)==3 and len(stride)==3)
        kernel_size2d1 = kernel_size[-2:]
        stride2d1 = stride[-2:]
        kernel_size2d2 = (kernel_size[0],kernel_size[0])
        stride2d2 = (kernel_size[0], stride[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=kernel_size2d1, stride=stride2d1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=kernel_size2d2, stride=stride2d2)
    def forward(self,x):
        x = self.maxpool1(x)
        x = x.transpose(1,3)
        x = self.maxpool2(x)
        x = x.transpose(1,3)
        return x 

class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
    def forward(self, x):
        return self.block(x)

class BackboneBlock(nn.Module):
    def __init__(self, ch_in, ch_out, dropout_rate=0.2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=64, kernel_size=3, stride=1), # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            small_basic_block(ch_in=64, ch_out=128),    # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # Replaced problematic maxpool_3d
            small_basic_block(ch_in=128, ch_out=256),   # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=256),   # *** 11 ***
            nn.BatchNorm2d(num_features=256),   # 12
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 14
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 4), stride=1, padding=(0, 1)),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=ch_out, kernel_size=(3, 1), stride=1, padding=(1, 0)), # 20
            nn.BatchNorm2d(num_features=ch_out),
            nn.ReLU(),  # *** 22 ***
        )

    def forward(self, x):
        out = self.backbone(x)
        #print(out.shape)
        return out

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        # [3, 112, 224] -> [4, 106, 218]
        self.conv1 = nn.Conv2d(3, 4, kernel_size=7, stride=1, padding=0)

        # [4, 106, 218] -> [8, 53, 109]
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1)

        # [8, 53, 109] -> [16, 27, 55]
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)

        # [16, 27, 55] -> [32, 14, 28]
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)

        # forzar a 32x32
        self.adjust = nn.Upsample(size=(32, 32), mode="bilinear", align_corners=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adjust(x)
        return x

# Modelo completo LSTM CustomBackbone
class OCRModel_LSTM_CustomBackbone(nn.Module):
    def __init__(self, vocab_size=64, hidden_size=512, num_layers=2, max_len=32):
        super().__init__()
        #self.mobilenet= mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        #self.mobilenet.classifier = nn.Identity()  # Eliminar la capa de clasificación
        self.backbone = BackboneBlock(ch_in=3, ch_out=256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #for param in self.backbone.parameters():
        #    param.requires_grad = False # Congelar los pesos del backbone
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.dim_features = 256
        self.adjust_dim = nn.Linear(self.dim_features, self.max_len*self.dim_features)
        self.lstm = nn.LSTM(
            self.dim_features,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        # Clasificador final: logits para cada carácter
        self.fc = nn.Linear(hidden_size * 2, self.vocab_size)

    def forward(self, x):
        feats = self.backbone(x)  # (N, C=256, H=50, W=200) -->(N, 256, 1, 1)
        feats = self.avgpool(feats).flatten(1)  # -->(N, 256)
        feats = self.adjust_dim(feats)  # -->(N, max_len * 256)
        feats = feats.view(-1, self.max_len, self.dim_features)  # -->(N, max_len, 256)
        # 2.  Pasar por LSTM
        out, _ = self.lstm(feats)  # -->(N, max_len, hidden_size)
        # 3. Logits para cada paso de la secuencia
        logits = self.fc(out)  # -->(N, max_len , vocab_size)
        return logits

# Modelo completo GRU CustomBackbone
class OCRModel_GRU_CustomBackbone(nn.Module):
    def __init__(self, vocab_size=64, hidden_size=512, num_layers=2, max_len=32):
        super().__init__()
        #self.mobilenet= mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        #self.mobilenet.classifier = nn.Identity()  # Eliminar la capa de clasificación
        self.backbone = BackboneBlock(ch_in=3, ch_out=256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.dim_features = 256
        self.adjust_dim = nn.Linear(self.dim_features, self.max_len*self.dim_features)
        # BiLSTM para modelar secuencia de caracteres
        self.lstm = nn.GRU(
            self.dim_features,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        # Clasificador final: logits para cada carácter
        self.fc = nn.Linear(hidden_size, self.vocab_size)
    def forward(self, x):
        # 1. Extraer features con Backbone MobileNetV2
        feats = self.backbone(x)  # (N, C=256, H=50, W=200) -->(N, 256, 1, 1)
        feats = self.avgpool(feats).flatten(1)  # -->(N, 256)
        feats = self.adjust_dim(feats)  # -->(N, max_len * 256)
        feats = feats.view(-1, self.max_len, self.dim_features)  # -->(N, max_len, 256)
        # 2.  Pasar por LSTM
        out, _ = self.lstm(feats)  # -->(N, max_len, hidden_size)
        # 3. Logits para cada paso de la secuencia
        logits = self.fc(out)  # -->(N, max_len , vocab_size)
        return logits

# Modelo completo LSTM mobilenetV3
class OCRModel_LSTM_MobileNet(nn.Module):
    def __init__(self, vocab_size=64, hidden_size=512, num_layers=2, max_len=32):
        super().__init__()
        self.mobilenet= mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        self.mobilenet.classifier = nn.Identity()  # Eliminar la capa de clasificación
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #for param in self.mobilenet.parameters():
        #    param.requires_grad = False # Congelar los pesos del backbone
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.dim_features = 960  # MobileNetV3-Large output features
        self.adjust_dim = nn.Linear(self.dim_features, max_len*256)
        self.lstm = nn.LSTM(
            256,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        # Clasificador final: logits para cada carácter
        self.fc = nn.Linear(hidden_size * 2, self.vocab_size)

    def forward(self, x):
        # 1. Extraer features con Backbone MobileNetV3
        feats = self.mobilenet.features(x)  # (N, 960, H/32, W/32)
        feats = self.avgpool(feats).flatten(1)  # -->(N, 960)
        feats = self.adjust_dim(feats)  # -->(N, max_len * 256)
        feats = feats.view(-1, self.max_len, 256)  # -->(N, max_len, 256)
        # 2.  Pasar por LSTM
        out, _ = self.lstm(feats)  # -->(N, max_len, hidden_size*2)
        # 3. Logits para cada paso de la secuencia
        logits = self.fc(out)  # -->(N, max_len , vocab_size)
        return logits

# Modelo completo GRU mobilenetV3
class OCRModel_GRU_MobileNet(nn.Module):
    def __init__(self, vocab_size=64, hidden_size=512, num_layers=2, max_len=32):
        super().__init__()
        self.mobilenet= mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        self.mobilenet.classifier = nn.Identity()  # Eliminar la capa de clasificación
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.dim_features = 960  # MobileNetV3-Large output features
        self.adjust_dim = nn.Linear(self.dim_features, max_len*256)
        # GRU para modelar secuencia de caracteres
        self.gru = nn.GRU(
            256,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        # Clasificador final: logits para cada carácter
        self.fc = nn.Linear(hidden_size, self.vocab_size)
    def forward(self, x):
        # 1. Extraer features con Backbone MobileNetV3
        feats = self.mobilenet.features(x)  # (N, 960, H/32, W/32)
        feats = self.avgpool(feats).flatten(1)  # -->(N, 960)
        feats = self.adjust_dim(feats)  # -->(N, max_len * 256)
        feats = feats.view(-1, self.max_len, 256)  # -->(N, max_len, 256)
        # 2.  Pasar por GRU
        out, _ = self.gru(feats)  # -->(N, max_len, hidden_size)
        # 3. Logits para cada paso de la secuencia
        logits = self.fc(out)  # -->(N, max_len , vocab_size)
        return logits


# Modelo completo Transformer
class OCRModel_Transformer(nn.Module):
    def __init__(self, vocab_size=64, max_len=32, hidden_dim=512, num_layers=2, num_heads=8, sos_idx=1):
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

# 🔹 Prueba
if __name__ == "__main__":
   x = torch.randn(4, 3, 50, 200)  # batch=4
   net = OCRModel_Transformer()
   y = net(x)
   print(y.shape, "Transformer")  # torch.Size([4, 32, 64])