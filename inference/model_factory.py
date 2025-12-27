from models.cnn import EEG_CNN
from models.vit import EEG_ViT
from models.resnet18 import EEG_ResNet18
from models.eegconvnext import EEGConvNeXt
from models.cnn_vit import CNN_ViT
from models.resnet18_vit import ResNet18_ViT
from models.eegconvnext_vit import EEGConvNeXt_ViT


def create_model(model_family):
    if model_family == "CNN":
        return EEG_CNN(in_ch=19, num_classes=3)

    elif model_family == "Vision Transformer":
        return EEG_ViT(num_classes=3)

    elif model_family == "ResNet-18":
        return EEG_ResNet18(num_classes=3)

    elif model_family == "EEGConvNeXt":
        return EEGConvNeXt(num_classes=3)

    elif model_family == "CNN + ViT":
        return CNN_ViT(num_classes=3)

    elif model_family == "ResNet-18 + ViT":
        return ResNet18_ViT(num_classes=3)

    elif model_family == "EEGConvNeXt + ViT":
        return EEGConvNeXt_ViT(num_classes=3)

    else:
        raise ValueError(f"Unknown model family: {model_family}")