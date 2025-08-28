import pandas as pd #library to read csv files, pd is alias word
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
ds=pd.read_csv("ehr_dataset.csv") #reading
print("=== First 5 values ===")
print(ds.head()) #print dataset headers

# Dataset Description and Info
print("=== Dataset Info ===")
print(ds.info())
print("Shape (rows, columns):", ds.shape)
print("="*80)

# Descriptive Statistics
print("=== Numeric Statistics ===")
print(ds.describe().T)
print("="*80)

print("data set summary with null rows")
print(ds.isnull().sum()) #checking number of null rows
ds = ds.dropna() #dropping all null values
print("="*80)

print("number of duplicate rows: ", ds.duplicated().sum()) #checking number of duplicate rows
ds = ds.drop_duplicates() #dropping duplicates if any
print("="*80)

#-----
numeric_cols = numeric_cols = ds.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    Q1 = ds[col].quantile(0.25) #.quartile() is a fuction, 0.25-> Q1
    Q3 = ds[col].quantile(0.75) #.quartile() is a fuction, 0.75-> Q2
    IQR = Q3 - Q1 #IQR
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    before = ds.shape[0] #row count before
    ds = ds[(ds[col] >= lower_bound) & (ds[col] <= upper_bound)] #keeping only the values in ds within the bounds
    after = ds.shape[0] #row count after
    print(f"{col}: lower={lower_bound}, upper={upper_bound}, min={ds[col].min()}, max={ds[col].max()}")
    print("="*80)


if 'Visit Date' in ds.columns: #checking if Visit_date in dataset.
    ds['Visit Date'] = pd.to_datetime(ds['Visit Date'], errors='coerce') #converting to standard format
    print("Standardized 'Visit Date' to datetime")
    
if 'Gender' in ds.columns:
    ds['Gender'] = ds['Gender'].str.strip().str.lower()   #remove spaces & make lowercase
    ds['Gender'] = ds['Gender'].map({'male': 'M', 'female': 'F'})
    print("Standardized 'Gender' column to M/F")
    
print("=== Convert Object Columns to Categorical ===")
cat_cols = ['Sex', 'Diagnosis', 'ICD-10 Code', 'Treatment', 'Comorbidity', 'Clinical Outcome']
for col in cat_cols:
    if col in ds.columns:
        ds[col] = ds[col].astype('category')
        print(f"Column '{col}' converted to category with {ds[col].nunique()} categories.")
        print(f"Sample categories for '{col}': {ds[col].unique()[:10]}")
print("="*80)


#Normalize Numeric Columns (Min-Max Scaler)
print("=== Normalizing Numeric Columns Using MinMaxScaler ===")
scaler = MinMaxScaler()
num_cols_to_scale = [col for col in numeric_cols if col in ds.columns]
if num_cols_to_scale:
    ds[num_cols_to_scale] = scaler.fit_transform(ds[num_cols_to_scale])
    print("Normalized columns:", num_cols_to_scale)
    print(ds[num_cols_to_scale].head())
print("="*80)


#Final Dataset Preview
ds = ds.round(2) #rounding off all normalized values to 2 decimals
print("=== Final Cleaned Data Shape ===")
print(ds.shape)
print("=== First 5 Rows of Cleaned Data ===")
print(ds.head())
print("="*80)



ds.to_csv("ehr_dataset_cleaned.csv", index=False) #to remove row no. index value -> index=false