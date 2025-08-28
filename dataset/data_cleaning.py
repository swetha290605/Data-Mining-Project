import pandas as pd #library to read csv files, pd is alias word
from sklearn.preprocessing import MinMaxScaler
ds=pd.read_csv("ehr_dataset.csv") #reading
print(ds.head()) #print dataset headers
print(ds.info()) #print dataset values
print(ds.describe()) #print dataset statistics

print("data set summary with null rows")
print(ds.isnull().sum()) #checking number of null rows
ds = ds.dropna() #dropping all null values

print("number of duplicate rows: ", ds.duplicated().sum()) #checking number of duplicate rows
ds = ds.drop_duplicates() #dropping duplicates if any

#-----
numeric_cols = ['Age','Systolic BP','Diastolic BP','Heart Rate','Temperature','Glucose','Cholesterol']
for col in numeric_cols:
    Q1 = ds[col].quantile(0.25) #.quartile() is a fuction, 0.25-> Q1
    Q3 = ds[col].quantile(0.75) #.quartile() is a fuction, 0.75-> Q2
    IQR = Q3 - Q1 #IQR
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    before = ds.shape[0] #row count before
    ds = ds[(ds[col] >= lower_bound) & (ds[col] <= upper_bound)] #keeping only the values in ds within the bounds
    after = ds.shape[0] #row count after
    print(f"Column '{col}': Removed {before - after} outliers")

if 'Visit Date' in ds.columns: #checking if Visit_date in dataset.
    ds['Visit Date'] = pd.to_datetime(ds['Visit Date'], errors='coerce') #converting to standard format
    print("Standardized 'Visit Date' to datetime")
if 'Gender' in ds.columns:
    ds['Gender'] = ds['Gender'].str.strip().str.lower()   # remove spaces & make lowercase
    ds['Gender'] = ds['Gender'].map({'male': 'M', 'female': 'F'})
    print("Standardized 'Gender' column to M/F")

cat_cols = ['Sex', 'Diagnosis', 'ICD-10 Code', 'Treatment', 'Comorbidity', 'Clinical Outcome'] 
for col in cat_cols: 
    if col in ds.columns: 
        ds[col] = ds[col].astype('category') #converting all non numeric objects to category data type for memory efficiency

scale = MinMaxScaler() #normalization using MinMax module according to formulae for optimization
num_cols_to_scale = [col for col in numeric_cols if col in ds.columns]
if num_cols_to_scale:
    ds[num_cols_to_scale] = scale.fit_transform(ds[num_cols_to_scale])
    print(f"Normalized columns: {num_cols_to_scale}")
    print(ds[num_cols_to_scale].head())




