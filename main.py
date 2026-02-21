import pandas as pd 

df=pd.read_csv("data/student-mat.csv")

print(df.head())

print("\nDataset Info:\n")
print(df.info())
