from load_wav import load_wav_as_array
from plot_spectral_centroid import plot_spectral_centroid
import os

input_folder = r"/Users/hanskjostelsen/AI - Folder/train_cut/engine3_good"
output_folder = r"/Users/hanskjostelsen/AI - Folder/Spectral_diagrams-good"

for file_name in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file_name)

    waveform, sr = load_wav_as_array(file_path)
    print("Loaded waveform:", waveform.shape, "Sample rate:", sr)

    output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_centroid.png")

    plot_spectral_centroid(file_path, save_path=output_file)
    print(f"Saved spectral centroid for {file_name} to {output_file}")