# ================================================================
# 📊 ELECTRONIC HEALTH RECORD DIAGNOSTIC SYSTEM (RANDOM FOREST, FINAL FIXED)
# ================================================================

import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

plt.style.use('ggplot')
sns.set_style('whitegrid')
warnings.filterwarnings("ignore")

# === 1. LOAD AND CLEAN DATA ===
try:
    ds = pd.read_csv("ehr_dataset.csv")
except FileNotFoundError:
    try:
        ds = pd.read_csv("ehr_dataset_cleaned.csv")
    except FileNotFoundError:
        print("❌ ERROR: Dataset file not found. Ensure 'ehr_dataset.csv' or 'ehr_dataset_cleaned.csv' exists.")
        sys.exit(1)

# Handle missing values safely for all types
for col in ds.columns:
    try:
        if isinstance(ds[col].dtype, pd.CategoricalDtype):
            if 'Missing' not in ds[col].cat.categories:
                ds[col] = ds[col].cat.add_categories(['Missing'])
            ds[col] = ds[col].fillna('Missing')
        else:
            ds[col] = ds[col].fillna('Missing')
    except Exception as e:
        print(f"⚠️ Skipping {col}: {e}")

print("✅ Dataset loaded successfully.\n")
print("=== Dataset Info ===")
ds.info()
print("\nShape:", ds.shape)
print("\nNull counts:\n", ds.isnull().sum())

# Drop duplicates only (keep missing values handled)
ds.drop_duplicates(inplace=True)

# Outlier Removal (IQR method)
numeric_cols = ds.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    Q1, Q3 = ds[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    ds = ds[(ds[col] >= Q1 - 1.5 * IQR) & (ds[col] <= Q3 + 1.5 * IQR)]

# Gender Cleanup
if 'Sex' in ds.columns:
    ds['Sex'] = ds['Sex'].astype(str).str.strip().str.lower().map({
        'male': 'M', 'female': 'F', 'm': 'M', 'f': 'F'
    })

# Convert categorical columns to category dtype
for col in ['Sex', 'Diagnosis', 'Treatment', 'Comorbidity', 'Clinical Outcome']:
    if col in ds.columns:
        ds[col] = ds[col].astype('category')

# === 1a. STATISTICAL DESCRIPTION ===
print("\n=== Statistical Description of Numeric Columns ===")
print(ds[numeric_cols].describe().T)

# Normalize numeric columns
scaler = MinMaxScaler()
if len(numeric_cols) > 0:
    ds_raw = ds[numeric_cols].copy()
    ds[numeric_cols] = scaler.fit_transform(ds[numeric_cols])

ds = ds.round(2)
ds.to_csv("ehr_dataset_cleaned.csv", index=False)
print("\n💾 Cleaned dataset saved as ehr_dataset_cleaned.csv\n")

# === 2. VISUALIZATION ===
numeric_cols_for_plot = [col for col in numeric_cols if col in ds.columns]
if numeric_cols_for_plot:
    n = len(numeric_cols_for_plot)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    plt.figure(figsize=(5 * ncols, 5 * nrows))
    for i, col in enumerate(numeric_cols_for_plot):
        plt.subplot(nrows, ncols, i + 1)
        sns.boxplot(y=ds[col], color='skyblue')
        plt.title(col)
    plt.tight_layout()
    plt.savefig('box_plots.png')
    plt.close()
    print("📈 Box plots saved as box_plots.png")

if len(numeric_cols_for_plot) > 1:
    sns.pairplot(ds[numeric_cols_for_plot])
    plt.savefig('scatter_plots.png')
    plt.close()
    print("📉 Scatter plots saved as scatter_plots.png")

# === 3. DEFINE RANGES ===
ranges_input = {
    "Systolic BP": [(100, 126, "Low"), (126, 153, "Medium"), (153, 180, "High")],
    "Diastolic BP": [(60, 76, "Low"), (76, 93, "Medium"), (93, 110, "High")],
    "Heart Rate": [(55, 77, "Low"), (77, 99, "Medium"), (99, 120, "High")],
    "Temperature": [(36.0, 37.1, "Low"), (37.1, 38.3, "Medium"), (38.3, 39.5, "High")],
    "Glucose": [(70, 112, "Low"), (112, 156, "Medium"), (156, 200, "High")],
    "Cholesterol": [(130, 180, "Low"), (180, 230, "Medium"), (230, 280, "High")],
    "Age": [(18, 30, "Young"), (30, 50, "Middle-aged"), (50, 80, "Elderly")]
}

def classify_value(value, measure):
    for low, high, label in ranges_input[measure]:
        if low <= value < high:
            return label
    return 'Missing'

# === 4. PREP DATA FOR RANDOM FOREST ===
bins_map = {}
for col in numeric_cols:
    bins = ds_raw[col].quantile([0, 0.33, 0.66, 1.0]).unique()
    labels = ['Low', 'Medium', 'High'][:len(bins) - 1]
    bins_map[col] = (bins, labels)
    ds[col + '_Bin'] = pd.cut(ds_raw[col], bins=bins, labels=labels, include_lowest=True)

feature_cols = [col + '_Bin' for col in numeric_cols] + ['Sex', 'Treatment', 'Comorbidity', 'Clinical Outcome']
feature_cols = [c for c in feature_cols if c in ds.columns]
ds_model = ds[feature_cols + ['Diagnosis']].copy()

# --- FIX: Safely add 'Missing' to categorical columns ---
for col in ds_model.select_dtypes(include=['category']).columns:
    if 'Missing' not in ds_model[col].cat.categories:
        ds_model[col] = ds_model[col].cat.add_categories(['Missing'])
    ds_model[col] = ds_model[col].fillna('Missing')

for col in ds_model.select_dtypes(include=['object']).columns:
    ds_model[col] = ds_model[col].fillna('Missing')

# Encode categorical variables
cat_cols = ds_model.select_dtypes(['category', 'object']).columns.drop('Diagnosis')
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    ds_model[col] = le.fit_transform(ds_model[col])
    le_dict[col] = le

target_le = LabelEncoder()
ds_model['Diagnosis'] = target_le.fit_transform(ds_model['Diagnosis'])

X = ds_model.drop(columns=['Diagnosis'])
y = ds_model['Diagnosis']

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# === 5. INTERACTIVE INPUT ===
print("\n=== PATIENT INPUT ===")
patient_features = {}

# Numeric inputs with displayed ranges
for measure, range_list in ranges_input.items():
    while True:
        ranges_text = " | ".join([f"{r[2]} ({r[0]}–{r[1]})" for r in range_list])
        val = input(f"Enter {measure} value ({ranges_text}): ")
        try:
            val = float(val)
            category = classify_value(val, measure)
            print(f"-> Classified as: {category}\n")
            patient_features[measure + '_Bin'] = category
            break
        except ValueError:
            print("⚠️ Invalid input. Please enter a number.\n")

# === 5b. IMPROVED CATEGORICAL INPUT WITH OPTIONS ===
for col in ['Sex', 'Treatment', 'Comorbidity', 'Clinical Outcome']:
    if col in ds_model.columns:
        options = sorted(ds[col].dropna().unique().tolist())
        print(f"\nAvailable options for {col}: {', '.join(map(str, options))}")
        val = input(f"Enter {col} (choose from above or type 'Missing' if unknown): ").strip()
        if val == '':
            val = 'Missing'
        elif val not in options and val != 'Missing':
            print(f"⚠️ '{val}' not found in options. Using 'Missing'.")
            val = 'Missing'
        patient_features[col] = val

# === 6. PREDICT FUNCTION (TOP 3 DIAGNOSES, FIXED) ===
def predict_patient(features_dict, top_n=3):
    df = pd.DataFrame([features_dict])

    # Encode categorical features
    for col in cat_cols:
        if col in df.columns:
            le = le_dict[col]
            if df[col][0] not in le.classes_:
                df[col][0] = 'Missing' if 'Missing' in le.classes_ else le.classes_[0]
            df[col] = le.transform(df[col])

    # Add missing columns and align feature order
    for col in X.columns:
        if col not in df.columns:
            df[col] = 0
    df = df[X.columns]

    # Predict probabilities
    probs = clf.predict_proba(df)[0]
    diagnoses = target_le.inverse_transform(np.argsort(probs)[::-1])
    top_probs = np.sort(probs)[::-1]
    top_confidences = [int(round(p * 10)) for p in top_probs]

    return list(zip(diagnoses[:top_n], top_confidences[:top_n]))

# === 7. OUTPUT ===
results = predict_patient(patient_features, top_n=3)

print("\n=== DIAGNOSIS RESULTS (TOP PREDICTIONS) ===")
for i, (diagnosis, confidence) in enumerate(results, start=1):
    print(f"{i}. 🩺 {diagnosis} — Confidence: {confidence}/10")

print("\n📋 Features Used:")
for k, v in patient_features.items():
    print(f"  - {k}: {v}")
