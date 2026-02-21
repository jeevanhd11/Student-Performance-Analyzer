import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("data/student-mat.csv")

print(df.head())

print("\nDataset Info:\n")
print(df.info())

plt.figure(figsize=(8,5))
sns.histplot(df['G3'], bins=20, kde=True)
plt.title("Distribution of Final Grades (G3)")
plt.xlabel("Final Grade")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x=df['studytime'], y=df['G3'])
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("Final Grade")
plt.show()

plt.figure(figsize=(12,8))
numeric_df = df.select_dtypes(include=['int64'])
sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()