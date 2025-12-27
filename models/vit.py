import torch
import torch.nn as nn

# ==================== SIMPLE VIT ====================
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


# ==================== VIT MODEL ====================
class EEG_ViT(nn.Module):
    """
    Input : (B, 19, H, W)
    Output: (B, num_classes)
    """
    def __init__(self, in_ch=19, num_classes=3, vit_dim=128, patch_size=4):
        super().__init__()

        # Patch Embedding (giữ nguyên H, W nếu patch_size=1)
        self.patch_embed = nn.Conv2d(
            in_ch, vit_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.vit = SimpleViT(
            dim=vit_dim,
            depth=2,
            heads=2,
            mlp_dim=256
        )

        self.norm = nn.LayerNorm(vit_dim)
        self.head = nn.Linear(vit_dim, num_classes)

    def forward(self, x):
        # x: (B, 19, H, W)
        x = self.patch_embed(x)            # (B, C, H', W')
        B, C, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)   # (B, N, C)
        x = self.vit(x)                    # (B, N, C)

        x = x.mean(dim=1)                  # Global average pooling
        x = self.norm(x)
        return self.head(x)