import os
import shutil


base_path = r"C:\Users\Gabriel\Documents\5. Semester\Machine_Lerning\Project"
name_file_path = os.path.join(base_path, "creating_dataset_heartbeat", "abnormal_d")
source_folder = os.path.join(base_path, "classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0", "training-d")
destination_folder = os.path.join(base_path, "creating_dataset_heartbeat", "test_heart_d", "abnormal")

os.makedirs(destination_folder, exist_ok=True)


with open(name_file_path, "r") as f:
    filenames = [line.strip() for line in f if line.strip()]


for name in filenames:
    wav_file = f"{name}.wav"
    src_path = os.path.join(source_folder, wav_file)
    dst_path = os.path.join(destination_folder, wav_file)

    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        print(f"Kopiert: {wav_file}")
    else:
        print(f"Nicht gefunden: {wav_file}")


