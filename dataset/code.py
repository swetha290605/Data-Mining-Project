import pandas as pd #library to read csv files, pd is alias word
ds=pd.read_csv("ehr_dataset.csv") #reading
print(ds.head()) #print dataset headers
print(ds.info()) #print dataset values
print(ds.describe()) #print dataset statistics