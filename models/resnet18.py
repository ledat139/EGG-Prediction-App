import torch
import torch.nn as nn
from torchvision.models import resnet18


class EEG_ResNet18(nn.Module):
    def __init__(self, in_ch=19, num_classes=3):
        super().__init__()

        # Load ResNet-18
        self.resnet = resnet18(weights=None)

        # Sửa conv1 để nhận số kênh tự do (mặc định 3)
        self.resnet.conv1 = nn.Conv2d(
            in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Thay FC cuối bằng số lớp
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnet(x)