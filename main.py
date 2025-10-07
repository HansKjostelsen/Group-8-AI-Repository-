from load_wav import load_wav_as_array
from plot_spectogram import plot_spectogram
import os

input_folder = r"C:\Users\Gabriel\Documents\5. Semester\Machine_Lerning\Project\train_cut\engine1_good"
output_folder = r"C:\Users\Gabriel\Documents\5. Semester\Machine_Lerning\Project\spectograms"

for file_name in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file_name)


    waveform, sr = load_wav_as_array(file_path)
    print("Loaded waveform:", waveform.shape, "Sample rate:", sr)

    # Generate output path for the spectrogram
    output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.png")
        
    # Plot and save spectrogram
    plot_spectogram(file_path, save_path=output_file)
    print(f"Saved spectrogram for {file_name} to {output_file}")


