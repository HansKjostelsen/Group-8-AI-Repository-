import os
import librosa
import numpy as np
import pandas as pd

BASE_PATH = r"C:\Users\Gabriel\Documents\5. Semester\Machine_Lerning\Project"
SUBFOLDER = "test_cut"
MFCC_COUNT = 20 

OUTPUT_CSV = "features_test.csv"

STATES = {
     "engine1_good": 0,
    "engine2_broken": 1,
    "engine3_heavyload": 2
}





def feature_extraction(file_path, mfcc_count=20):
    
    y, sr = librosa.load(file_path, sr=None, mono=True)
    #y = audio time series
    #sr = the sampling rate of the audio

    #this is where we start to extract several features from the wav file

    # 1) MFCC
    mfcc_mean = np.mean(
        librosa.feature.mfcc(y=y, sr=sr, n_mfcc=mfcc_count),
        axis=1
    )
    #mfccs_mean gives us a 1D array of average values for each MFCC coefficent
    
     # 2) Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_bw_mean = np.mean(spec_bw)

    # 3) Spectral Centroid
    spec_centroid_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    # 4) Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # 5) Zero Crossing Rate
    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y))

    # 6) RMS Energy
    rms_mean = np.mean(librosa.feature.rms(y=y))

    return np.hstack([
    mfcc_mean,               # 20 values
    spec_bw_mean,            # 1 value
    spec_centroid_mean,      # 1 value
    chroma_mean,             # 12 values
    zcr_mean,                # 1 value
    rms_mean                 # 1 value
])




def main():
    #forming the full path directory
    split_dir = os.path.join(BASE_PATH, SUBFOLDER)
    all_rows = []

    for state, label in STATES.items():
        state_dir = os.path.join(split_dir, state)

        wav_files = sorted(
            f for f in os.listdir(state_dir) if f.endswith(".wav")
        )
        #just making sure we have only wav files and they are in a proper order
        for filename in wav_files:
            path = os.path.join(state_dir, filename)
            features = feature_extraction(path, MFCC_COUNT)

            # append features + label as one row
            all_rows.append(np.append(features, label))

    if not all_rows:
        print("No data extracted.")
        return
    #this is where we build the Dataframe, by dynamically construction columns based on how many features were extracted (definded in feature_extraction)
    columns = (
    [f"mfcc_{i+1}" for i in range(MFCC_COUNT)] +
    ["spec_bw", "spec_centroid"] +
    [f"chroma_{i+1}" for i in range(12)] +
    ["zcr", "rms", "label"]
)

    df = pd.DataFrame(all_rows, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Feature extraction saved to {OUTPUT_CSV}")
    print("Saved at:", os.path.abspath(OUTPUT_CSV))


if __name__ == "__main__":
    main()

