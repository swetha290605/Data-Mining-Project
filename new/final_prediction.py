from data_cleaning import *
from classifier_evaluation import *
# 5. Input
print("\n=== PATIENT INPUT ===")
patient_features = {}

# Numeric Inputs with displayed ranges
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
            print("Invalid input. Please enter a number.\n")

# Category Inputs with displayed options
for col in ['Sex', 'Treatment', 'Comorbidity', 'Clinical Outcome']:
    if col in ds_model.columns:
        options = sorted(ds[col].dropna().unique().tolist())
        print(f"\nAvailable options for {col}: {', '.join(map(str, options))}")
        val = input(f"Enter {col} (choose from above or type 'Missing' if unknown): ").strip()
        if val == '':
            val = 'Missing'
        elif val not in options and val != 'Missing':
            print(f"'{val}' not found in options. Using 'Missing'.")
            val = 'Missing'
        patient_features[col] = val

# === 6. Prediction
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
    probs = rf.predict_proba(df)[0]
    diagnoses = target_le.inverse_transform(np.argsort(probs)[::-1])
    top_probs = np.sort(probs)[::-1]
    top_confidences = [int(round(p * 10)) for p in top_probs]

    return list(zip(diagnoses[:top_n], top_confidences[:top_n]))

# 7. Output
results = predict_patient(patient_features, top_n=3)

print("\n=== DIAGNOSIS RESULTS (TOP PREDICTIONS) ===")
for i, (diagnosis, confidence) in enumerate(results, start=1):
    print(f"{i}. {diagnosis} — Confidence: {confidence}/10")

print("\nFeatures Used:")
for k, v in patient_features.items():
    print(f"  - {k}: {v}")

