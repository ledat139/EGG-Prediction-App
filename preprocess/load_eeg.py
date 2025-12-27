import mne
import numpy as np
import pywt
from skimage.transform import resize
from joblib import Parallel, delayed
import gc

# ================= CONFIG =================
SEGMENT_LENGTH_SEC = 4
SAMPLING_RATE = 500
OVERLAP_RATIO = 0.5

SEGMENT_LENGTH = SEGMENT_LENGTH_SEC * SAMPLING_RATE
OVERLAP_STEP = int(SEGMENT_LENGTH * (1 - OVERLAP_RATIO))

# ---- CWT parameters (đúng như khóa luận) ----
CWT_FREQ_MIN = 0.5
CWT_FREQ_MAX = 45
CWT_N_FREQS = 65

# ==========================================

def compute_cwt_spectrogram(signal_1d, fs):
    """
    Tính CWT spectrogram cho 1 kênh EEG
    Output shape: (65, T)
    """
    freqs = np.linspace(CWT_FREQ_MIN, CWT_FREQ_MAX, CWT_N_FREQS)
    scales = (1.0 * fs) / freqs

    coeffs, _ = pywt.cwt(
        signal_1d,
        scales,
        'morl',
        sampling_period=1 / fs
    )

    power = np.abs(coeffs) ** 2
    power_db = 10 * np.log10(power + 1e-8)

    return power_db.astype(np.float32)


def load_eeg(
    eeg_set_path: str,
    overlap_ratio: float = 0.5,
    n_jobs: int = -1,          # dùng toàn bộ core
    resize_for_model: bool = True
):
    """
    ONE-PASS EEG preprocessing:
    - CWT chạy 1 lần
    - Preview + Predict dùng chung kết quả
    - Multicore theo channel
    """

    raw = mne.io.read_raw_eeglab(eeg_set_path, preload=True, verbose=False)
    data = raw.get_data()
    n_channels, n_samples = data.shape

    overlap_step = int(SEGMENT_LENGTH * (1 - overlap_ratio))

    segments = []
    preview_cwt = None

    start = 0
    seg_idx = 0

    while start + SEGMENT_LENGTH <= n_samples:
        segment = data[:, start:start + SEGMENT_LENGTH]

        # ==== MULTICORE CWT theo channel ====
        channel_specs = Parallel(n_jobs=n_jobs)(
            delayed(compute_cwt_spectrogram)(
                segment[ch], SAMPLING_RATE
            )
            for ch in range(n_channels)
        )

        # Resize cho model (trừ preview nếu muốn giữ nguyên)
        if resize_for_model:
            channel_specs = [
                resize(
                    Sxx, (65, 224),
                    order=1,              # nhanh hơn order=3
                    mode="reflect",
                    anti_aliasing=True
                ).astype(np.float32)
                for Sxx in channel_specs
            ]

        tensor = np.stack(channel_specs, axis=0)
        segments.append(tensor)

        # Lưu preview segment đầu
        if seg_idx == 0:
            preview_cwt = tensor.copy()

        start += overlap_step
        seg_idx += 1

    segments = np.stack(segments)

    del raw
    gc.collect()

    return segments, data, preview_cwt

