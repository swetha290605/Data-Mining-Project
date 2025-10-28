from data_cleaning import *
X = ds_model.drop(columns=['Diagnosis'])
y = ds_model['Diagnosis']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3, random_state=42)
y_pred=rf.predict(X_test)
con_mat=confusion_matrix(y_test,y_pred)
print("Confusion Matrix (Random Forest):\n", con_mat)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report (Random Forest):\n", classification_report(y_test, y_pred))

svm=SVC(kernel="rbf", random_state=42)
svm.fit(X_train, y_train)
y_pred_svm=svm.predict(X_test)
print("Confusion Matrix (SVM):\n", confusion_matrix(y_test, y_pred_svm))
print("\nSVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report (SVM):\n", classification_report(y_test, y_pred_svm))

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_log = lr.predict(X_test)
print("Confusion Matrix (Logistic Regression):\n", confusion_matrix(y_test, y_pred_log))
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("Classification Report (Logistic Regression):\n", classification_report(y_test, y_pred_log))

models = {
    'Random Forest': rf,
    'SVM': svm,
    'Logistic Regression': lr
}

results = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append({'Model': name, 'Accuracy': acc})

results_df = pd.DataFrame(results)
print("\nComparing Models:\n", results_df)
best_model = results.loc[results['Accuracy'].idxmax(), 'Model']
print(f"\nHence {best_model} is the best classifier")