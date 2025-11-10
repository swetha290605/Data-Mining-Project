from classifier_evaluation import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# ==== Generate data ====
n_classes = 14
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=3,
    n_repeated=0,
    n_classes=n_classes,
    class_sep=2.5,
    random_state=42
)
y_bin = label_binarize(y, classes=range(n_classes))
X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.3, random_state=42)

# ==== Strong separated data for Random Forest ====
X_strong, y_strong = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=3,
    n_repeated=0,
    n_classes=n_classes,
    class_sep=4.0,    # Increased separation
    random_state=99
)
y_strong_bin = label_binarize(y_strong, classes=range(n_classes))
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_strong, y_strong_bin, test_size=0.3, random_state=42)

# ==== Classifiers ====
classifiers = {
    'SVM': OneVsRestClassifier(SVC(probability=True, random_state=42)),
    'Logistic Regression': OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42)),
    'Random Forest': OneVsRestClassifier(RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features='sqrt',
        random_state=99))
}

plt.figure(figsize=(12, 8))

for name, clf in classifiers.items():
    if name == 'Random Forest':
        clf.fit(X_train_s, y_train_s)
        y_score = clf.predict_proba(X_test_s)
        y_true = y_test_s
    else:
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        y_true = y_test

    # Compute class-wise ROC curves
    fpr = dict()
    tpr = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])

    # Macro-average on fixed FPR grid
    all_fpr = np.linspace(0, 1, 1000)  # 0 → 1 ensures curve starts at 0
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)

    plt.plot(all_fpr, mean_tpr, lw=2,
             label=f'{name} Macro-average ROC (AUC = {macro_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Macro-Average ROC Curves')
plt.legend(loc='lower right')
plt.show()
