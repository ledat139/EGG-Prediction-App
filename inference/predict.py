import torch
import numpy as np
from inference.loader import load_models

CLASS_MAP = {0: "A", 1: "F", 2: "C"}


@torch.no_grad()
def predict_with_voting(segments, model_family):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = load_models(model_family, device)

    segments = torch.tensor(segments).float().to(device)

    model_votes = []

    for model in models:
        logits = model(segments)
        preds = logits.argmax(dim=1).cpu().numpy()

        unique, counts = np.unique(preds, return_counts=True)
        model_votes.append(unique[np.argmax(counts)])

    unique, counts = np.unique(model_votes, return_counts=True)
    final_pred = unique[np.argmax(counts)]

    vote_detail = {CLASS_MAP[k]: int(v) for k, v in zip(unique, counts)}

    return CLASS_MAP[final_pred], vote_detail