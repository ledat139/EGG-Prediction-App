import torch.nn as nn
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, 4*dim)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4*dim, dim)
    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = x.permute(0,2,3,1)
        x = self.norm(x)
        x = self.pw1(x); x = self.act(x); x = self.pw2(x)
        x = x.permute(0,3,1,2)
        return x + identity

class SimpleViT(nn.Module):
    def __init__(self, dim, depth=2, heads=2, mlp_dim=256):
        super().__init__()
        layer = nn.TransformerEncoderLayer(dim, heads, mlp_dim, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
    def forward(self, x): return self.encoder(x)

class EEGConvNeXt_ViT(nn.Module):
    def __init__(self, in_ch=19, num_classes=3, vit_dim=128):
        super().__init__()
        dims = [32,64,128,256]; depths=[1,1,2,1]
        self.stem = nn.Sequential(nn.Conv2d(in_ch,dims[0],3,padding=1), nn.BatchNorm2d(dims[0]))
        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(*[ConvNeXtBlock(dims[i]) for _ in range(depths[i])]))
            if i<3: self.stages.append(nn.Conv2d(dims[i],dims[i+1],3,stride=2,padding=1))
        self.proj = nn.Conv2d(dims[-1], vit_dim, 1)
        self.vit = SimpleViT(dim=vit_dim, depth=2, heads=2, mlp_dim=256)
        self.head = nn.Sequential(nn.LayerNorm(vit_dim), nn.Linear(vit_dim,num_classes))
    def forward(self,x):
        x = self.stem(x)
        for s in self.stages: x = s(x)
        x = self.proj(x)
        B,C,H,W = x.shape
        x = x.reshape(B,C,H*W).permute(0,2,1)
        x = self.vit(x)
        x = x.mean(1)
        return self.head(x)