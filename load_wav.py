import numpy as np
import wave

def load_wav_as_array (file_path):
    with wave.open(file_path, "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        audio_array = np.frombuffer(frames, dtype = np.int16)
        sample_rate = wav_file.getframerate()
    return audio_array, sample_rate