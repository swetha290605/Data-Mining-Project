# --- Import everything needed ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from classifier_evaluation import *


X, y = make_classification(
    n_samples=1000, n_features=20, n_classes=2, random_state=42
)

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --- Scale features for SVM and Logistic Regression ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Define models ---
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}

plt.figure(figsize=(8, 6))

# --- Detect binary or multiclass ---
n_classes = len(np.unique(y))
print(f"Detected number of classes: {n_classes}")

best_auc = 0
best_model = ""
auc_scores = {}

# --- Train models & plot ROC ---
for name, model in models.items():
    model.fit(X_train, y_train)

    if n_classes == 2:
        # Binary classification
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

    else:
        # Multiclass classification
        y_test_bin = label_binarize(y_test, classes=np.unique(y))
        y_score = model.predict_proba(X_test)
        roc_auc = 0
        for i in range(n_classes):
            fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc += auc(fpr_i, tpr_i)
        roc_auc /= n_classes  # Average AUC across all classes

    auc_scores[name] = roc_auc
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    if roc_auc > best_auc:
        best_auc = roc_auc
        best_model = name

# --- Plot formatting ---
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()

# --- Save figure ---
plt.savefig("roc_curves.png")
plt.show()

# --- Print AUC summary ---
print("\nROC AUC Comparison:")
for name, score in auc_scores.items():
    print(f"{name}: AUC = {score:.3f}")

print(f"\n🏆 Best Model: {best_model}")
print("ROC saved as: roc_curves.png")
