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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Copy dataset
df_model = df.copy()

# Encode categorical columns
label_encoder = LabelEncoder()

for column in df_model.select_dtypes(include=['object']).columns:
    df_model[column] = label_encoder.fit_transform(df_model[column])
    X = df_model.drop("G3", axis=1)
    y = df_model["G3"] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")
