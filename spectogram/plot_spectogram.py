import matplotlib.pyplot as plt
from load_wav import load_wav_as_array
import numpy as np
from scipy.signal import spectrogram


def plot_spectogram(file_path, save_path=None, save_image = False):
    waveform, sr = load_wav_as_array(file_path)
    if waveform.ndim > 1:
        waveform = waveform[:, 0]

    # Compute spectrogram
    f, t, Sxx = spectrogram(waveform, fs=sr, nperseg=1024, noverlap=512)
    Sxx_log = 10 * np.log10(Sxx + 1e-10)  # convert to dB scale

    return Sxx_log, f, t, sr