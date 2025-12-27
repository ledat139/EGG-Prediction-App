import torch.nn as nn
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, 4*dim)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4*dim, dim)
    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = x.permute(0,2,3,1)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = x.permute(0,3,1,2)
        return x + identity

class EEGConvNeXt(nn.Module):
    def __init__(self, in_ch=19, num_classes=3):
        super().__init__()
        dims = [32, 64, 128, 256]
        depths = [1,1,2,1]   # giảm để train nhanh hơn
        self.stem = nn.Sequential(nn.Conv2d(in_ch, dims[0], 3, padding=1), nn.BatchNorm2d(dims[0]))
        self.stages = nn.ModuleList()
        for i in range(4):
            blocks = [ConvNeXtBlock(dims[i]) for _ in range(depths[i])]
            stage = nn.Sequential(*blocks)
            self.stages.append(stage)
            if i < 3:
                self.stages.append(nn.Conv2d(dims[i], dims[i+1], 3, stride=2, padding=1))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(dims[-1], num_classes))
    def forward(self, x):
        x = self.stem(x)
        for s in self.stages:
            x = s(x)
        return self.head(x)