Performing heartbeat sound classification using machine learning models trained on audio features extracted directly from .wav files.

The goal is to distinguish between normal and abnormal heart sounds.



Dataset information can be found under the folder:

creating\_dataset\_heartbeat/



*feature\_extraction\_heart.py*



Loads each .wav file from the organized dataset (normal / abnormal folders).



Extracts the following audio features:



MFCCs (20 coefficients)



Spectral Bandwidth



Spectral Centroid



Chroma Features



Zero Crossing Rate



RMS Energy



Combines all extracted features into a single feature vector.



Saves the resulting feature vectors and labels into a CSV file for training and evaluation.

