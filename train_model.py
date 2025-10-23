import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Make sure Python can find compute_dataset.py in the same folder
#original problem that the folder could not be found is fixed this way
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from compute_dataset import load_spectogram_dataset



#Normalization
#The goal was to normalize every spectrum individually.
#The first approach was to normalize in generaly, but the model performance was only about 30%

def normalize_spectrograms(X):#X is a 3D NumPy array containing multiple spectograms, where every spectogram is a 2D array -> shape (num_samples, height, width)
    """Normalize each spectrogram individually to [0, 1]."""
    X = X.astype("float32")#here we convert the array to the type fload32
    #why float32? -> original dataset stored as int (0-255). Since the division between int values can lead to rounding errors, float32 is used to allow for smooth scaling
    X_min = np.min(X, axis=(1, 2), keepdims=True)#here we find the minimum value within each spectogram. 
    #why axis = (1,2) -> we want to find the minimum across height and width, while keeping the index axis
    #why  keepdims=True -> without it NumPy  would remove the reduced dimensions
    X_max = np.max(X, axis=(1, 2), keepdims=True) #same ideas for the maximum as for the minimum
    X_norm = (X - X_min) / (X_max - X_min + 1e-8) #the smales value becomes zero and the largest becomes 1. We add the small constant to avoid dividing by zero
    return X_norm

#Why do we normalize in the first place?
#normalization is one of the most critical steps in the preprocessing of data.
#models can have strange behaviour when the input values have larger ranges
#also
#frequencies for example which have large numerica ranges would dominate other features if we didn't normalize




# Data Preparation

def prepare_data(train_folder, test_folder):

    # Load training data

    X_train, y_train = load_spectogram_dataset(train_folder)
    X_train = normalize_spectrograms(X_train)
    X_train = X_train.reshape((X_train.shape[0], -1))  #this converts every 2D spectogram into a 1D feature vector
    #why? X_train.shape([0], -1) -> with the -1 you let NumPy automatically calculate this dimension

    # Load test data

    X_test, y_test = load_spectogram_dataset(test_folder)
    X_test = normalize_spectrograms(X_test)
    X_test = X_test.reshape((X_test.shape[0], -1))
    #same ideas for test data as for train data

    return X_train, X_test, y_train, y_test



# Model Training

def train_random_forest(X_train, y_train, n_estimators=200):
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        class_weight="balanced"  # handles class imbalance automatically
    )
    clf.fit(X_train, y_train)
    return clf


def train_svm(X_train, y_train, kernel="rbf", C=1.0):
    clf = SVC(kernel=kernel, C=C, probability=True, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    return clf


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_model(clf, X_test, y_test, model_name="Model"):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{model_name} - Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.show()

    # Feature importance (nur für Random Forest)
    if isinstance(clf, RandomForestClassifier):
        importances = clf.feature_importances_
        plt.figure(figsize=(6, 3))
        plt.plot(np.sort(importances)[::-1][:50])
        plt.title("Top 50 Feature Importances (Random Forest)")
        plt.xlabel("Feature Rank")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.show()


# -----------------------------
# Main Pipeline
# -----------------------------
def build_and_train_model(train_folder, test_folder, model_type="random_forest"):
    X_train, X_test, y_train, y_test = prepare_data(train_folder, test_folder)

    if model_type == "random_forest":
        clf = train_random_forest(X_train, y_train)
        model_name = "Random Forest"
    elif model_type == "svm":
        clf = train_svm(X_train, y_train)
        model_name = "SVM"
    else:
        raise ValueError("model_type must be 'random_forest' or 'svm'")

    evaluate_model(clf, X_test, y_test, model_name)

    # Save model for later use
    model_path = f"{model_type}_model.joblib"
    joblib.dump(clf, model_path)
    print(f"{model_name} saved as '{model_path}'")

    return clf


# -----------------------------
# Run script
# -----------------------------
if __name__ == "__main__":
    train_folder = r"C:\Users\Gabriel\Documents\5. Semester\Machine_Lerning\Project\dataset\train"
    test_folder = r"C:\Users\Gabriel\Documents\5. Semester\Machine_Lerning\Project\dataset\test"

    print("Training Random Forest...")
    rf_model = build_and_train_model(train_folder, test_folder, model_type="random_forest")

    print("\nTraining SVM...")
    svm_model = build_and_train_model(train_folder, test_folder, model_type="svm")
