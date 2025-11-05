import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


train_df = pd.read_csv("features_train_a.csv")
test_df = pd.read_csv("features_test_b.csv")


X_train = train_df.drop("label", axis=1).values
y_train = train_df["label"].values

X_test = test_df.drop("label", axis=1).values
y_test = test_df["label"].values


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

#Train SVM

svm_model = SVC(kernel="rbf", C=10, gamma=0.01, random_state=42)
svm_model.fit(X_train_scaled, y_train)

svm_preds = svm_model.predict(X_test_scaled)

print("===== SVM Results =====")
print("Accuracy:", accuracy_score(y_test, svm_preds))
print("Classification Report:")
print(classification_report(y_test, svm_preds))
plot_confusion_matrix(y_test, svm_preds, "SVM Confusion Matrix")

# ======== Optional: Cross-Validation for SVM =========
from sklearn.model_selection import cross_val_score
scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5)
print(f"Mean Cross-Validation Score: {scores.mean()}")

# ======== Optional: Check Overfitting by spliting train set for SVM =========
from sklearn.model_selection import train_test_split
X_subtrain, X_valid, y_subtrain, y_valid = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)
svm_model.fit(X_subtrain, y_subtrain)
print("Overfitting check:", svm_model.score(X_valid, y_valid))

#Train Random Forest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train_scaled, y_train)

rf_preds = rf_model.predict(X_test_scaled)

print("\n===== Random Forest Results =====")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print("Classification Report:")
print(classification_report(y_test, rf_preds))
plot_confusion_matrix(y_test, rf_preds, "Random Forest Confusion Matrix")


