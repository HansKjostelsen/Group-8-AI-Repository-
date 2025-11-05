import numpy as np
import os
import wave
import matplotlib.pyplot as plt

# === Define your file paths ===
good_path = r"C:\Users\Gabriel\Documents\5. Semester\Machine_Lerning\Project\train_cut\engine1_good\pure_0.wav"
broken_path = r"C:\Users\Gabriel\Documents\5. Semester\Machine_Lerning\Project\train_cut\engine2_broken\pure_0.wav"
heavyload_path = r"C:\Users\Gabriel\Documents\5. Semester\Machine_Lerning\Project\train_cut\engine3_heavyload\pure_0.wav"

# === Function to read a .wav file into a numpy array ===
def read_wav_as_array(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        audio_array = np.frombuffer(frames, dtype=np.int16)
    return audio_array

# === Read the three signals ===
audio_good = read_wav_as_array(good_path)
audio_broken = read_wav_as_array(broken_path)
audio_heavy = read_wav_as_array(heavyload_path)

# === Plot all three on one figure ===
plt.figure(figsize=(10, 5))
plt.plot(audio_good, label='Good Motor', color='green', alpha=0.7)
plt.plot(audio_broken, label='Broken Motor', color='red', alpha=0.5)
plt.plot(audio_heavy, label='Heavyload Motor', color='blue', alpha=0.4)

plt.title("Motor Sound Comparison")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.show()
