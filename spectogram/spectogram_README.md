The spectogram folder focuses on generating spectogram-based Features from audio data, 

storing them as NumPy Arrays and Training machine learning models (RFC and SVM) using these features



**File Disriptions:**



*creating\_npy\_dataset.py*

Converts raw audio samples (.wav files) into NumPy array files (.npy format).



*plot\_spectogram.py*

visualizes the spectogram for inspection and Analysis



*compute\_dataset.py*

Iterates through all available samples in .npy form and returns the dataset in a ready-to-train format (X any y Arrays)



*train\_model.py*

main training script 

calls compute\_dataset to load the prepared spectogram dataset, trains svm, rfc and evaluates model Performance



