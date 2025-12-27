import torch
import numpy as np
from inference.loader import load_models

CLASS_MAP = {0: "A", 1: "F", 2: "C"}

def predict_with_voting(eeg, model_family):
    models = load_models(model_family)

    votes = []

    for model in models:
        model.eval()
        with torch.no_grad():
            x = torch.tensor(eeg).float().unsqueeze(0)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).item()
            votes.append(pred)

    # Voting
    unique, counts = np.unique(votes, return_counts=True)
    final_pred = unique[np.argmax(counts)]

    vote_detail = {
        CLASS_MAP[k]: int(v)
        for k, v in zip(unique, counts)
    }

    return CLASS_MAP[final_pred], vote_detail
