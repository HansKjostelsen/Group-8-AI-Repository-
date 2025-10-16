from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import numpy as np
from load_wav import load_wav_as_array


def compute_spectral_centroid(waveform, sr):
    # Hvis stereo → ta én kanal
    if waveform.ndim > 1:
        waveform = waveform[:, 0]
    
    # Beregn STFT / spekter
    freqs, times, Sxx = spectrogram(waveform, fs=sr, nperseg=1024, noverlap=512)
    
    # Unngå log0 ved å legge på veldig lite tall
    magnitude = np.abs(Sxx) + 1e-10

    # Formelen for spectral centroid (vektet sum av frekvensbånd)
    centroid = np.sum(freqs[:, None] * magnitude, axis=0) / np.sum(magnitude, axis=0)

    return times, centroid

def plot_spectral_centroid(file_path, save_path=None):
    waveform, sr = load_wav_as_array(file_path)
    times, centroid = compute_spectral_centroid(waveform, sr)

    plt.figure(figsize=(10, 4))
    plt.plot(times, centroid)
    plt.title("Spectral Centroid")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()



