import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

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



param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [0.001, 0.01, 0.1, 1],
    "kernel": ["rbf"]
}

grid = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=1,   # ✅ Fix: no multiprocessing
    verbose=2
)

grid.fit(X_train_scaled, y_train)

print("Best parameters:", grid.best_params_)
print("Best cross-validation accuracy:", grid.best_score_)

# Train and evaluate best model on test set
best_svm = grid.best_estimator_
svm_preds = best_svm.predict(X_test_scaled)

print("\n===== Tuned SVM Results =====")
print("Accuracy:", accuracy_score(y_test, svm_preds))
print("Classification Report:")
print(classification_report(y_test, svm_preds))
plot_confusion_matrix(y_test, svm_preds, "Tuned SVM Confusion Matrix")














#train SVM 
#svm_model = SVC(kernel="rbf", C=10, gamma=0.01, class_weight='balanced', random_state=42)
#svm_model = SVC(kernel="rbf", C=10, gamma=0.01, random_state=42)
#svm_model.fit(X_train_scaled, y_train)

#svm_preds = svm_model.predict(X_test_scaled)

#print("===== SVM Results =====")
#print("Accuracy:", accuracy_score(y_test, svm_preds))
#print("Classification Report:")
#print(classification_report(y_test, svm_preds))
#plot_confusion_matrix(y_test, svm_preds, "SVM Confusion Matrix")



# ======== 2) Train Random Forest =========
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train_scaled, y_train)

rf_preds = rf_model.predict(X_test_scaled)

print("\n===== Random Forest Results =====")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print("Classification Report:")
print(classification_report(y_test, rf_preds))
plot_confusion_matrix(y_test, rf_preds, "Random Forest Confusion Matrix")



