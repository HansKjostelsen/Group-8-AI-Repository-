import os
import numpy as np
from plot_spectogram import plot_spectogram

input_folder = r"C:\Users\Gabriel\Documents\5. Semester\Machine_Lerning\Project\test_cut\engine3_heavyload"
output_folder = r"C:\Users\Gabriel\Documents\5. Semester\Machine_Lerning\Project\dataset\test\spectrograms_heavyload"

os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    if not file_name.endswith(".wav"):
        continue

    file_path = os.path.join(input_folder, file_name)
    print("Processing:", file_path)

    Sxx_log, f, t, sr = plot_spectogram(file_path, save_image=True,
                                            save_path=os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.png"))

    # Save data as numpy array
    np.save(os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.npy"), Sxx_log)

    print(f"Saved: {file_name}")


