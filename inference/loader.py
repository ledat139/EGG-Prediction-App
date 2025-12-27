from concurrent.futures import ThreadPoolExecutor
import os
import torch
from inference.model_factory import create_model

MODEL_DIR_MAP = {
    "CNN": "checkpoints/cnn",
    "Vision Transformer": "checkpoints/vit",
    "ResNet-18": "checkpoints/resnet18",
    "EEGConvNeXt": "checkpoints/eegconvnext",
    "CNN + ViT": "checkpoints/cnn_vit",
    "ResNet-18 + ViT": "checkpoints/resnet18_vit",
    "EEGConvNeXt + ViT": "checkpoints/eegconvnext_vit"
}

def load_one_model(file_path, model_family, device):
    model = create_model(model_family)
    state_dict = torch.load(file_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def load_models(model_family, device="cpu"):
    model_dir = MODEL_DIR_MAP[model_family]
    model_files = sorted(f for f in os.listdir(model_dir) if f.endswith(".pth"))
    model_paths = [os.path.join(model_dir, f) for f in model_files]

    models = []
    with ThreadPoolExecutor(max_workers=len(model_paths)) as executor:
        results = executor.map(lambda p: load_one_model(p, model_family, device), model_paths)
        models = list(results)

    return models
