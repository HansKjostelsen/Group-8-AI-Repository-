import matplotlib.pyplot as plt
from load_wav import load_wav_as_array
import numpy as np

def plot_spectogram(file_path, save_path=None):
    waveform, sr = load_wav_as_array(file_path)
    
    # Optional: if stereo, take just one channel
    if waveform.ndim > 1:
        waveform = waveform[:, 0]

    plt.figure(figsize=(10, 4))
    plt.specgram(waveform, Fs=sr, NFFT=1024, noverlap=512, cmap='inferno')
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(label="Intensity [dB]")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()