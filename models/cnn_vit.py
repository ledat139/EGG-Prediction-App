import torch
import torch.nn as nn


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
    def __init__(self, in_ch, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=1)

    def forward(self, x):
        x = self.proj(x)                  # (B, D, H, W)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x
        
class CNN_ViT(nn.Module):
    def __init__(self, in_ch=19, num_classes=3, vit_dim=128, dropout=0.4):
        super().__init__()

        # ===== 1. CNN Backbone (GIỐNG BASELINE) =====
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # ===== 2. Patch embedding =====
        self.patch_embed = PatchEmbed(
            in_ch=256,
            embed_dim=vit_dim
        )

        # ===== 3. Positional embedding =====
        self.pos_embed = nn.Parameter(torch.randn(1, 512, vit_dim))

        # ===== 4. Simple ViT =====
        self.vit = SimpleViT(
            dim=vit_dim,
            depth=2,
            heads=2,
            mlp_dim=256
        )

        # ===== 5. Classification head =====
        self.head = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Dropout(dropout),
            nn.Linear(vit_dim, num_classes, bias=True)
        )

    def forward(self, x):
        x = self.features(x)        # (B, 256, H', W')
        x = self.patch_embed(x)     # (B, N, D)

        N = x.size(1)
        x = x + self.pos_embed[:, :N, :]

        x = self.vit(x)
        x = x.mean(dim=1)

        return self.head(x)