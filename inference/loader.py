import torch
import os

MODEL_DIR_MAP = {
    "EEGConvNeXt + ViT": "models/eegconvnext_vit",
    "EEGConvNeXt": "models/eegconvnext",
    "ResNet-18 + ViT": "models/resnet18_vit",
    "ResNet-18": "models/resnet18",
    "CNN + ViT": "models/cnn_vit",
    "CNN": "models/cnn",
    "ViT": "models/vit"
}

def load_models(model_family):
    model_dir = MODEL_DIR_MAP[model_family]
    model_files = sorted(os.listdir(model_dir))

    models = []
    for file in model_files:
        if file.endswith(".pth"):
            model = torch.load(
                os.path.join(model_dir, file),
                map_location="cpu"
            )
            model.eval()
            models.append(model)

    return models
