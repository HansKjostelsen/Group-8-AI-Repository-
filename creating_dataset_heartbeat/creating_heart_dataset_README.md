This script processes the PhysioNet/cinC Challenge 2016 dataset and builds a structured Folder hierachy for heartbeat Audio classification.

The goal is to extract the correct .wav files from the original dataset and organize them in a way similar to the existing motor dataset structure.



**Source of original Dataset:**

https://physionet.org/content/challenge-2016/1.0.0/



*sorting.py*

the PhysioNet dataset provides multiple training and test sets (e.g., training-a, training-b, test-d, etc.).

Each folder contains many .wav audio recordings and corresponding label files.

However, the audio files are not stored in labeled subfolders (e.g., “normal” or “abnormal”), so this script was created to extract and organize them automatically.

