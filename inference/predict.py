import torch
import numpy as np
from inference.loader import load_models
import gc
CLASS_MAP = {0: "A", 1: "F", 2: "C"}

import torch
import numpy as np
from inference.loader import load_models

CLASS_MAP = {0: "A", 1: "F", 2: "C"}

@torch.no_grad()
def predict_with_voting(segments, model_family):
    """
    segments: np.array, shape (n_segments, n_channels, height, width)
    model_family: str
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load tất cả model + mean/std
    model_tuples = load_models(model_family, device)

    model_votes = []

    for model, mean, std in model_tuples:
        # Chuyển segments sang tensor và normalize riêng theo model
        x = torch.tensor(segments).float()
        mean_tensor = torch.tensor(mean[:, None, None]).float()
        std_tensor = torch.tensor(std[:, None, None]).float()
        x = (x - mean_tensor) / (std_tensor + 1e-6)
        x = x.to(device)

        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()

        # Voting theo segment
        unique, counts = np.unique(preds, return_counts=True)
        model_votes.append(unique[np.argmax(counts)])

    # Voting cuối cùng theo model
    unique, counts = np.unique(model_votes, return_counts=True)
    final_pred = unique[np.argmax(counts)]

    vote_detail = {CLASS_MAP[k]: int(v) for k, v in zip(unique, counts)}
    del model_tuples
    torch.cuda.empty_cache()
    gc.collect()
    return CLASS_MAP[final_pred], vote_detail
