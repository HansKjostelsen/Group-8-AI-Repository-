Performing engine Sound classification using machine learning models trained on Audio Features extracted directly from .wav files



*feature\_extraction.py*

loads each .wav file and extracts the following Features:MFCC (20 values), spectral bandwidth, spectral centroid, chroma Features, Zero Crossing rate and RMS Energy



combines all Features into a single feature Vector and saves them into a CSV file 



*train\_models.py*

loads training and testing feature CSV files

splits the data into Features (X) and labels (y)

scales all Features using StandardScaler

trains SVM and RFC

evaluates the model

