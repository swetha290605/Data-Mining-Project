from data_cleaning import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd
import numpy as np

# Load data
X = ds_model.drop(columns=['Diagnosis'])
y = ds_model['Diagnosis']

# Train-test split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=42)

def calc_sensitivity_specificity(cm, labels):
    results = []
    total = cm.sum()
    for i, label in enumerate(labels):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = total - (TP + FP + FN)
        sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
        results.append({
            'Class': label,
            'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
            'Sensitivity': sensitivity,
            'Specificity': specificity
        })
    return pd.DataFrame(results)

# Train Random Forest on full X and y for consistency
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)  # You had this, but better keep fit only on train generally

# Predict on test
y_pred = rf.predict(X_test)

# Confusion matrix
con_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (Random Forest):\n", con_mat)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report (Random Forest):\n", classification_report(y_test, y_pred))
df_rf = calc_sensitivity_specificity(con_mat, labels)
df_rf['Model'] = 'Random Forest'
print("\nRandom Forest Sensitivity and Specificity per Class:")
print(df_rf[['Class', 'Sensitivity', 'Specificity']])

# Function to extract TP, TN, FP, FN per class from n x n confusion matrix
def get_classwise_metrics(cm, labels):
    for i, label in enumerate(labels):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)
        print(f"Class '{label}': TP={TP}, TN={TN}, FP={FP}, FN={FN}")

# Get labels sorted
labels = sorted(np.unique(np.concatenate((y_test, y_pred))))

# Print TP, TN, FP, FN per diagnosis class
print("\nPer-Class Metrics:")
get_classwise_metrics(con_mat, labels)

# Repeat for SVM
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
con_mat_svm=confusion_matrix(y_test, y_pred_svm)
print("Confusion Matrix (SVM):\n", confusion_matrix(y_test, y_pred_svm))
print("\nSVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report (SVM):\n", classification_report(y_test, y_pred_svm))
df_svm = calc_sensitivity_specificity(con_mat_svm, labels)
df_svm['Model'] = 'SVM'
print("\nSVM Sensitivity and Specificity per Class:")
print(df_svm[['Class', 'Sensitivity', 'Specificity']])
get_classwise_metrics(con_mat_svm, labels)


lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_log = lr.predict(X_test)
con_mat_lr=confusion_matrix(y_test, y_pred_log)
print("Confusion Matrix (Logistic Regression):\n", confusion_matrix(y_test, y_pred_log))
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("Classification Report (Logistic Regression):\n", classification_report(y_test, y_pred_log))
df_lr = calc_sensitivity_specificity(con_mat_lr, labels)
df_lr['Model'] = 'Logistic Regression'
print("\nLogistic Regression Sensitivity and Specificity per Class:")
print(df_lr[['Class', 'Sensitivity', 'Specificity']])
get_classwise_metrics(con_mat_lr, labels)

final_df = pd.concat([df_rf, df_svm, df_lr], ignore_index=True)
print(final_df)

models = {'Random Forest': rf, 'SVM': svm, 'Logistic Regression': lr}
results = []
for name, model in models.items():
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results.append({'Model': name, 'Accuracy': acc})
results_df = pd.DataFrame(results)
print("\nComparing Models:\n", results_df)
best_model = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
print(f"\nHence {best_model} is the best classifier")
