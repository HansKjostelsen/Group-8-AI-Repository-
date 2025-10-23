import os
import numpy as np

def load_spectogram_dataset(base_folder):
    X, y = [], [] #inialize two empty python lists

    labels = ["good", "broken", "heavy_load"] # create a python list with the labels

    for label_index, label in enumerate(labels): #every label is assigned with an index
        folder = os.path.join(base_folder, label) #creating a full path to the subfolder
        for file in os.listdir(folder): #now we loop over all files in our subfolder
            if file.endswith(".npy"):
                data = np.load(os.path.join(folder, file)) #load every .npy file into memory as a NumPy array

                data = data[:128, :128] #makes every example the same size, because CNN often expects same dimensions

                X.append(data) #store spectogram data (2D NumPy array) in X. Here 
                y.append(label_index) #store numeric labels in y

    X = np.array(X) #now X is a 3D NumPy array (num_samples, 128, 128)
    y = np.array(y) #1D array of numeric labels
    return X , y 