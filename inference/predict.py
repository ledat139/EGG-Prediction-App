import torch
import numpy as np
from inference.loader import load_models

CLASS_MAP = {0: "A", 1: "F", 2: "C"}

@torch.no_grad()
def predict_with_voting(segments, model_family):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_tuples = load_models(model_family, device)

    result = {
        "model_votes": {},
        "final_vote": None
    }

    model_level_preds = []

    for idx, (model, mean, std) in enumerate(model_tuples):
        model_name = f"{model_family}_model_{idx+1}"

        x = torch.tensor(segments).float()
        mean_tensor = torch.tensor(mean[:, None, None]).float()
        std_tensor = torch.tensor(std[:, None, None]).float()
        x = (x - mean_tensor) / (std_tensor + 1e-6)
        x = x.to(device)

        logits = model(x)
        seg_preds = logits.argmax(dim=1).cpu().numpy()

        # segment-level count
        unique, counts = np.unique(seg_preds, return_counts=True)
        seg_count = {
            CLASS_MAP[int(k)]: int(v)
            for k, v in zip(unique, counts)
        }

        # model-level vote
        model_vote = unique[np.argmax(counts)]
        model_vote_label = CLASS_MAP[int(model_vote)]
        model_level_preds.append(model_vote)

        result["model_votes"][model_name] = {
            "segment_counts": seg_count,
            "model_vote": model_vote_label
        }

    # final voting giữa các model
    unique, counts = np.unique(model_level_preds, return_counts=True)
    final_pred = unique[np.argmax(counts)]
    result["final_vote"] = CLASS_MAP[int(final_pred)]

    return result

