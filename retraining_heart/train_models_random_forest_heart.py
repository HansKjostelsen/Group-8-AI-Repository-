

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

train_data = pd.read_csv("features_train_a.csv")
test_data = pd.read_csv("features_test_b.csv")

# Split features and labels
X_train = train_data.drop(columns=["label"]).values
y_train = train_data["label"].values

X_test = test_data.drop(columns=["label"]).values
y_test = test_data["label"].values


#Scaling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#Confusion Matrix Plot

def visualize_confusion(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


#Support Vector Machine (SVM)


svm_clf = SVC(kernel="rbf", C=10, gamma=0.01, random_state=42)
svm_clf.fit(X_train_scaled, y_train)

svm_predictions = svm_clf.predict(X_test_scaled)


print(f"Test Accuracy: {accuracy_score(y_test, svm_predictions):.4f}")
print("Classification Report:")
print(classification_report(y_test, svm_predictions))

visualize_confusion(y_test, svm_predictions, "SVM Confusion Matrix")

# Cross-validation performance
cv_scores = cross_val_score(svm_clf, X_train_scaled, y_train, cv=5)
print(f"Mean 5-Fold CV Accuracy: {cv_scores.mean():.4f}")

# Overfitting check using validation split
X_subtrain, X_val, y_subtrain, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)
svm_clf.fit(X_subtrain, y_subtrain)
val_score = svm_clf.score(X_val, y_val)
print(f"SVM Validation Accuracy (Overfitting Check): {val_score:.4f}")


#Random Forest Classifier


rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
rf_clf.fit(X_train_scaled, y_train)

rf_predictions = rf_clf.predict(X_test_scaled)

print("\n===== Random Forest Evaluation =====")
print(f"Test Accuracy: {accuracy_score(y_test, rf_predictions):.4f}")
print("Classification Report:")
print(classification_report(y_test, rf_predictions))

visualize_confusion(y_test, rf_predictions, "Random Forest Confusion Matrix")



