import os
import torch
import numpy as np
from inference.model_factory import create_model


MODEL_DIR_MAP = {
    "CNN": "checkpoints/cnn",
    "Vision Transformer": "checkpoints/vit",
    "ResNet-18": "checkpoints/resnet18",
    "EEGConvNeXt": "checkpoints/eegconvnext",
    "CNN + ViT": "checkpoints/cnn_vit",
    "ResNet-18 + ViT": "checkpoints/resnet18_vit",
    "EEGConvNeXt + ViT": "checkpoints/eegconvnext_vit",
}


def load_one_model(file_path: str, model_family: str, device: str = "cpu"):
    """
    Load 1 model + mean/std (CPU-only, safe for Streamlit)
    """
    # --- load model ---
    model = create_model(model_family)

    state_dict = torch.load(file_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # --- load mean / std ---
    prefix = os.path.splitext(os.path.basename(file_path))[0]
    base_dir = os.path.dirname(file_path)

    mean_path = os.path.join(base_dir, f"{prefix}_mean.npy")
    std_path = os.path.join(base_dir, f"{prefix}_std.npy")

    if not os.path.exists(mean_path):
        raise RuntimeError(f"Missing mean file: {mean_path}")
    if not os.path.exists(std_path):
        raise RuntimeError(f"Missing std file: {std_path}")

    mean = np.load(mean_path)
    std = np.load(std_path)
    std[std == 0] = 1.0  # tránh chia 0

    return model, mean, std


def load_models(model_family: str, device: str = "cpu"):
    """
    Load tất cả model trong folder (tuần tự, KHÔNG multi-thread)
    Trả về list [(model, mean, std), ...]
    """
    if model_family not in MODEL_DIR_MAP:
        raise RuntimeError(f"Unknown model family: {model_family}")

    model_dir = MODEL_DIR_MAP[model_family]

    if not os.path.exists(model_dir):
        raise RuntimeError(f"Model directory not found: {model_dir}")

    model_files = sorted(f for f in os.listdir(model_dir) if f.endswith(".pth"))

    if len(model_files) == 0:
        raise RuntimeError(f"No .pth model found in {model_dir}")

    models = []
    for fname in model_files:
        path = os.path.join(model_dir, fname)
        models.append(load_one_model(path, model_family, device))

    return models
