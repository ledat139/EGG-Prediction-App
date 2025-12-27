import numpy as np
import pandas as pd

def load_eeg(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
        eeg = df.values
    elif file.name.endswith(".npy"):
        eeg = np.load(file)
    else:
        raise ValueError("Unsupported EEG format")

    # Nếu nhiều channel → lấy trung bình (demo)
    if eeg.ndim == 2:
        eeg = eeg.mean(axis=1)

    return eeg.astype(np.float32)
