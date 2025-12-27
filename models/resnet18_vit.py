import torch
import torch.nn as nn
from torchvision.models import resnet18

class SimpleViT(nn.Module):
    def __init__(self, dim, depth=2, heads=2, mlp_dim=256):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x):
        return self.encoder(x)

class PatchEmbed(nn.Module):
    def __init__(self, in_ch, patch_size=1, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(
            in_ch, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)                   # (B, D, H, W)
        x = x.flatten(2).transpose(1, 2)   # (B, N, D)
        return x


class ResNet18_ViT(nn.Module):
    def __init__(self, in_ch=19, num_classes=3, vit_dim=128):
        super().__init__()

        # ===== 1. ResNet-18 Backbone =====
        resnet = resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(
            in_ch, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4        # (B, 512, H/32, W/32)
        )

        # ===== 2. Patch Embedding =====
        self.patch_embed = PatchEmbed(
            in_ch=512,
            patch_size=1,
            embed_dim=vit_dim
        )

        # ===== 3. Positional Embedding =====
        self.pos_embed = nn.Parameter(torch.randn(1, 1000, vit_dim))

        # ===== 4. ViT Encoder (GIỐNG CODE 2) =====
        self.vit = SimpleViT(
            dim=vit_dim,
            depth=2,
            heads=2,
            mlp_dim=256
        )

        # ===== 5. Classification Head =====
        self.head = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, num_classes)
        )

    def forward(self, x):
        # CNN
        x = self.backbone(x)

        # Patch → token
        x = self.patch_embed(x)
        B, N, D = x.shape

        # Positional embedding
        x = x + self.pos_embed[:, :N, :]

        # ViT
        x = self.vit(x)

        # Global average pooling (token-wise)
        x = x.mean(dim=1)

        return self.head(x)