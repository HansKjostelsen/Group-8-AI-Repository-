import os
import numpy as np

def load_spectogram_dataset(base_folder):
    X, y = [], []  # initialize empty lists for data and labels

    labels = ["normal", "abnormal"]  # your new heartbeat classes

    for label_index, label in enumerate(labels):
        folder = os.path.join(base_folder, label)

        if not os.path.exists(folder):
            print(f"Warning: folder '{folder}' not found, skipping.")
            continue

        for file in os.listdir(folder):
            if file.endswith(".npy"):
                data = np.load(os.path.join(folder, file))

                # Ensure consistent size — crop or pad to 128×128
                data = data[:128, :128]
                if data.shape[1] < 128:
                    pad_width = 128 - data.shape[1]
                    data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')

                X.append(data)
                y.append(label_index)

    X = np.array(X)
    y = np.array(y)

    print(f"Loaded {len(X)} samples from '{base_folder}'")
    return X, y