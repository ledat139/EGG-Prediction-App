import time
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
    num_segments = len(segments)

    # ================= TOTAL TIMER =================
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_total_start = time.perf_counter()

    print(f"\n[INFO] Start inference | segments={num_segments} | device={device}")

    for idx, (model, mean, std) in enumerate(model_tuples):
        model_name = f"{model_family}_model_{idx+1}"

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_model_start = time.perf_counter()

        x = torch.tensor(segments).float()
        mean_tensor = torch.tensor(mean[:, None, None]).float()
        std_tensor = torch.tensor(std[:, None, None]).float()
        x = (x - mean_tensor) / (std_tensor + 1e-6)
        x = x.to(device)

        logits = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_model_end = time.perf_counter()

        seg_preds = logits.argmax(dim=1).cpu().numpy()

        unique, counts = np.unique(seg_preds, return_counts=True)
        seg_count = {
            CLASS_MAP[int(k)]: int(v)
            for k, v in zip(unique, counts)
        }

        model_vote = unique[np.argmax(counts)]
        model_vote_label = CLASS_MAP[int(model_vote)]
        model_level_preds.append(model_vote)

        # ===== CONSOLE LOG =====
        print(
            f"[MODEL] {model_name:20s} | "
            f"time = {t_model_end - t_model_start:.4f} s"
        )

        result["model_votes"][model_name] = {
            "segment_counts": seg_count,
            "model_vote": model_vote_label
        }

    # ================= FINAL VOTING =================
    unique, counts = np.unique(model_level_preds, return_counts=True)
    final_pred = unique[np.argmax(counts)]
    result["final_vote"] = CLASS_MAP[int(final_pred)]

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_total_end = time.perf_counter()

    total_time = t_total_end - t_total_start
    avg_time_per_segment_ms = (total_time / num_segments) * 1000

    # ===== CONSOLE SUMMARY =====
    print("\n[SUMMARY]")
    print(f"  Total inference time : {total_time:.4f} s")
    print(f"  Avg time / segment   : {avg_time_per_segment_ms:.2f} ms")
    print(f"  Final prediction     : {result['final_vote']}")
    print("-" * 50)

    # 🔴 GIỮ NGUYÊN return
    return result
