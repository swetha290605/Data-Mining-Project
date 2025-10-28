# EHR system with Random Forest Method of Model Training

import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

plt.style.use('ggplot')
sns.set_style('whitegrid')
warnings.filterwarnings("ignore")

# 1. Load and Clean
try:
    ds = pd.read_csv("ehr_dataset.csv")
except FileNotFoundError:
    print("ERROR: Dataset file not found. Ensure 'ehr_dataset.csv' or 'ehr_dataset_cleaned.csv' exists.")
    sys.exit(1)

# missing values handling
for col in ds.columns:
    try:
        if isinstance(ds[col].dtype, pd.CategoricalDtype):
            if 'Missing' not in ds[col].cat.categories:
                ds[col] = ds[col].cat.add_categories(['Missing'])
            ds[col] = ds[col].fillna('Missing')
        else:
            ds[col] = ds[col].fillna('Missing')
    except Exception as e:
        print(f"Skipping {col}: {e}")

print("Dataset loaded successfully.\n")
print("------Dataset Info----- ")
ds.info()
print("\nShape:", ds.shape)
print("\nNull counts:\n", ds.isnull().sum())

# duplicates handling
ds.drop_duplicates(inplace=True)

# outlier handling using IQR
numeric_cols = ds.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    Q1, Q3 = ds[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    ds = ds[(ds[col] >= Q1 - 1.5 * IQR) & (ds[col] <= Q3 + 1.5 * IQR)]

# normalizing formats
if 'Sex' in ds.columns:
    ds['Sex'] = ds['Sex'].astype(str).str.strip().str.lower().map({
        'male': 'M', 'female': 'F', 'm': 'M', 'f': 'F'
    })

scaler = MinMaxScaler()
if len(numeric_cols) > 0:
    ds_raw = ds[numeric_cols].copy()
    ds[numeric_cols] = scaler.fit_transform(ds[numeric_cols])

ds = ds.round(2)
ds.to_csv("ehr_dataset_cleaned.csv", index=False)


# Convert categorical columns to category dtype
for col in ['Sex', 'Diagnosis', 'Treatment', 'Comorbidity', 'Clinical Outcome']:
    if col in ds.columns:
        ds[col] = ds[col].astype('category')

print("\nCleaned dataset saved as ehr_dataset_cleaned.csv\n")

# Displaying Data Statistics
print("\n-----Statistical Description of Numeric Columns -----")
print(ds[numeric_cols].describe().T)

# 2. Data Visualization 
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
    print(" Box plots saved as box_plots.png")

if len(numeric_cols_for_plot) > 1:
    sns.pairplot(ds[numeric_cols_for_plot])
    plt.savefig('scatter_plots.png')
    plt.close()
    print(" Scatter plots saved as scatter_plots.png")

# 3. Input range definition
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

# Random Forest Format Making
bins_map = {}
for col in numeric_cols:
    bins = ds_raw[col].quantile([0, 0.33, 0.66, 1.0]).unique()
    labels = ['Low', 'Medium', 'High'][:len(bins) - 1]
    bins_map[col] = (bins, labels)
    ds[col + '_Bin'] = pd.cut(ds_raw[col], bins=bins, labels=labels, include_lowest=True)

feature_cols = [col + '_Bin' for col in numeric_cols] + ['Sex', 'Treatment', 'Comorbidity', 'Clinical Outcome']
feature_cols = [c for c in feature_cols if c in ds.columns]
ds_model = ds[feature_cols + ['Diagnosis']].copy()

# Fixing
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
