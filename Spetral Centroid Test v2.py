#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Imports:
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

BASE = r"/Users/hanskjostelsen/AI - Folder"
CENTROID_TRAIN = os.path.join(BASE, "centroid_dataset", "train")
CENTROID_TEST  = os.path.join(BASE, "centroid_dataset", "test")


#%%
#Load dataset
def load_spectral_centroid_dataset(base_folder, fixed_length=128):
    X, y = [], []
    labels = ["good", "broken", "heavy_load"]

    for label_index, label in enumerate(labels):
        folder = os.path.join(base_folder, label)
        if not os.path.isdir(folder):
            print(f"Advarsel: fant ikke mappe {folder}")
            continue

        for file in os.listdir(folder):
            if not file.endswith(".npy"):
                continue

            data = np.load(os.path.join(folder, file))  

            
            if data.shape[0] >= fixed_length:
                data_fixed = data[:fixed_length]
            else:
                data_fixed = np.zeros(fixed_length, dtype=np.float32)
                data_fixed[:data.shape[0]] = data

            X.append(data_fixed)
            y.append(label_index)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y

#%%
#Load and combine:
    #Reads train and test data, and combines them into one dataset
def load_all_data():
    
    X_train, y_train = load_spectral_centroid_dataset(CENTROID_TRAIN)
    X_test,  y_test  = load_spectral_centroid_dataset(CENTROID_TEST)

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    print("Totalt datasett:", X.shape, y.shape)
    return X, y


#Feature extraction
def extract_features(X):
    
    feats = []
    for x in X:
        feats.append([
            np.mean(x),
            np.std(x),
            np.min(x),
            np.max(x),
            skew(x),
            kurtosis(x)
        ])
    return np.array(feats, dtype=np.float32)


# Data preperation:
def prepare_data(test_size=0.2, random_state=42):
    X, y = load_all_data()

    X = extract_features(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape, "y_test shape:", y_test.shape)

    return X_train, X_test, y_train, y_test, scaler


#%%
#Random Forrest train:
def train_random_forest(X_train, y_train):
    clf = RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf


#SVM train:
def train_svm(X_train, y_train):
    clf = SVC(
        kernel="rbf",
        C=50.0,
        gamma=0.03,      
        probability=True,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    return clf

#%%
#Evaluating:
def evaluate_model(clf, X_train, y_train, X_test, y_test, model_name="Model"):
    y_pred = clf.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred)
    y_pred_train = clf.predict(X_train)
    acc_train = accuracy_score(y_train, y_pred_train)

    print(f"\n{model_name}")
    print(f"Train Accuracy: {acc_train * 100:.2f}%")
    print(f"Test Accuracy:  {acc_test * 100:.2f}%\n")

    print("Classification Report (test):\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.show()

#%%
#Main:
def build_and_train_model(model_type="random_forest"):
    X_train, X_test, y_train, y_test, scaler = prepare_data()

    if model_type == "random_forest":
        clf = train_random_forest(X_train, y_train)
        model_name = "Random Forest (Centroid Stats, merged)"
    elif model_type == "svm":
        clf = train_svm(X_train, y_train)
        model_name = "SVM (Centroid Stats, merged)"
    else:
        raise ValueError("model_type must be 'random_forest' or 'svm'")

    evaluate_model(clf, X_train, y_train, X_test, y_test, model_name)

    
    joblib.dump({"model": clf, "scaler": scaler},
                f"{model_type}_centroid_merged_model.joblib")
    print(f"{model_name} saved.")

    return clf


if __name__ == "__main__":
    print("Training Random Forest...")
    rf_model = build_and_train_model(model_type="random_forest")

    print("\nTraining SVM...")
    svm_model = build_and_train_model(model_type="svm")
