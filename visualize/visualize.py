# visualize.py
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from preprocess.load_eeg import CWT_FREQ_MIN, CWT_FREQ_MAX, CWT_N_FREQS

# =========================
# Vẽ tín hiệu EEG thô
# =========================
def plot_raw_eeg(data, fs=500, n_channels=19, max_time_sec=5, ch_names=None):
    """
    Vẽ tín hiệu EEG thô (trước CWT)
    data: np.ndarray (n_channels, n_samples)
    ch_names: list[str], tên các kênh, default=None → dùng số
    """
    n_samples = data.shape[1]
    t = np.arange(n_samples) / fs
    n_plot = min(n_channels, data.shape[0])
    plt.figure(figsize=(12, 2*n_plot))

    for ch in range(n_plot):
        plt.subplot(n_plot, 1, ch+1)
        plt.plot(t[:fs*max_time_sec], data[ch, :fs*max_time_sec])
        title = ch_names[ch] if ch_names is not None else f"Channel {ch+1}"
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel("uV")
        plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()



# =========================
# Vẽ CWT spectrogram 1 kênh
# =========================
def plot_cwt_spectrogram(spectrogram, channel_name=None, figsize=(12,4)):
    """
    Vẽ spectrogram sau khi tính CWT cho 1 kênh
    spectrogram: np.ndarray (n_freqs, n_time)
    channel_name: str, tên kênh
    """
    plt.figure(figsize=figsize)
    plt.imshow(spectrogram, aspect='auto', origin='lower',
               extent=[0, spectrogram.shape[1], CWT_FREQ_MIN, CWT_FREQ_MAX],
               cmap='jet')
    plt.colorbar(label='Power [dB]')
    ch_name = channel_name if channel_name is not None else "Channel"
    plt.title(f"CWT Spectrogram - {ch_name}")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [samples]")
    st.pyplot(plt.gcf())
    plt.close()



# =========================
# Vẽ toàn bộ các kênh trong lưới
# =========================
def plot_cwt_grid(spectrograms, n_rows=5, n_cols=4, figsize=(15,12), ch_names=None):
    """
    Vẽ tất cả các kênh EEG của 1 segment dưới dạng grid
    spectrograms: np.ndarray (n_channels, n_freqs, n_time)
    ch_names: list[str] hoặc None
    """
    n_channels = spectrograms.shape[0]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for ch in range(n_channels):
        im = axes[ch].imshow(spectrograms[ch], aspect="auto", origin="lower",
                             extent=[0, spectrograms.shape[2], CWT_FREQ_MIN, CWT_FREQ_MAX],
                             cmap="magma")
        title = ch_names[ch] if ch_names is not None else f"Channel {ch+1}"
        axes[ch].set_title(title)
        axes[ch].set_xlabel("Time bins")
        axes[ch].set_ylabel("Freq bins")
        fig.colorbar(im, ax=axes[ch], fraction=0.046, pad=0.04)

    # Ẩn subplot thừa nếu có
    for ax in axes[n_channels:]:
        ax.axis("off")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

