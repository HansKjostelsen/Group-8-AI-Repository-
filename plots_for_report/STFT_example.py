import os
import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy.signal import stft  

good_folder_path = r"C:\Users\Gabriel\Documents\5. Semester\Machine_Lerning\Project\train_cut\engine2_broken"
file_name = "pure_12.wav"
my_wav_file = os.path.join(good_folder_path, file_name)

#Read .wav file
with wave.open(my_wav_file, 'rb') as wav_file:
    frames = wav_file.readframes(wav_file.getnframes())
    audio_array = np.frombuffer(frames, dtype=np.int16)
    sample_rate = wav_file.getframerate()  # get sampling frequency

print(f"Sample rate: {sample_rate} Hz")
print(f"Audio length: {len(audio_array) / sample_rate:.2f} s")

#Compute STFT
# nperseg controls the time-frequency resolution tradeoff
f, t, Zxx = stft(audio_array, fs=sample_rate, nperseg=1024)

#Plot
plt.figure(figsize=(10, 6))
plt.pcolormesh(f, t, np.abs(Zxx).T, shading='gouraud')
plt.title("STFT broken example")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Time [s]")
plt.colorbar(label="Amplitude")
plt.tight_layout()
plt.show()


