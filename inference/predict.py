import time
import torch
import numpy as np
from inference.loader import load_models
import gc

CLASS_MAP = {0: "AD", 1: "FTD", 2: "CN"}

@torch.no_grad()
def predict_with_voting(segments, model_family):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
      model_tuples = load_models(model_family, device)
    except:
      print("An exception occurred")

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
    del x
    del mean_tensor
    del std_tensor
    del logits
    del model_tuples
    del model_level_preds

    if device.type == "cuda":
        torch.cuda.empty_cache()

    gc.collect()
    return result

import time
import gc
import torch
import numpy as np
import os
from inference.model_factory import create_model
from inference.loader import MODEL_DIR_MAP

CLASS_MAP = {0: "AD", 1: "FTD", 2: "CN"}

@torch.no_grad()
def predict_with_streaming_ensemble(segments, model_family):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = MODEL_DIR_MAP[model_family]

    model_files = sorted(f for f in os.listdir(model_dir) if f.endswith(".pth"))

    if len(model_files) == 0:
        raise RuntimeError("No models found")

    num_segments = len(segments)
    model_level_preds = []
    model_votes_detail = {}

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_total_start = time.perf_counter()

    for idx, fname in enumerate(model_files):
        model_path = os.path.join(model_dir, fname)
        prefix = os.path.splitext(fname)[0]

        # ===== LOAD ONE MODEL =====
        model = create_model(model_family)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        mean = np.load(os.path.join(model_dir, f"{prefix}_mean.npy"))
        std = np.load(os.path.join(model_dir, f"{prefix}_std.npy"))
        std[std == 0] = 1.0

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_model_start = time.perf_counter()

        # ===== PREPROCESS =====
        x = torch.tensor(segments).float()
        mean_t = torch.tensor(mean[:, None, None]).float()
        std_t = torch.tensor(std[:, None, None]).float()
        x = (x - mean_t) / (std_t + 1e-6)
        x = x.to(device)

        logits = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_model_end = time.perf_counter()

        seg_preds = logits.argmax(dim=1).cpu().numpy()

        unique, counts = np.unique(seg_preds, return_counts=True)
        model_vote = unique[np.argmax(counts)]
        model_vote_label = CLASS_MAP[int(model_vote)]

        model_level_preds.append(model_vote)
        model_votes_detail[f"{model_family}_model_{idx+1}"] = {
            "segment_counts": {
                CLASS_MAP[int(k)]: int(v) for k, v in zip(unique, counts)
            },
            "model_vote": model_vote_label,
            "time": t_model_end - t_model_start
        }

        # ===== CLEAN MEMORY =====
        del model, state_dict, logits, x, mean_t, std_t, seg_preds
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # ===== FINAL VOTING =====
    unique, counts = np.unique(model_level_preds, return_counts=True)
    final_pred = unique[np.argmax(counts)]

    t_total_end = time.perf_counter()

    return {
        "model_votes": model_votes_detail,
        "final_vote": CLASS_MAP[int(final_pred)],
        "total_time": t_total_end - t_total_start
    }
