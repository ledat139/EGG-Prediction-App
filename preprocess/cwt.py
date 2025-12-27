import pywt
import numpy as np

def compute_cwt(signal, scales=None, wavelet="morl"):
    if scales is None:
        scales = np.arange(1, 64)

    coef, _ = pywt.cwt(signal, scales, wavelet)
    return coef
