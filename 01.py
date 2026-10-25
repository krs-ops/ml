import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv(r"C:\Users\ADMIN\Desktop\1st pract(ML)\Titanic.csv")

print("***Original Data:\n")
print(data.head())

print("\n\n***Missing Values:\n")
print(data.isnull().sum())

if "Age" in data.columns:
    imputer = SimpleImputer(strategy = "mean")
    data["Age"] = imputer.fit_transform(data[["Age"]])

if "Sex" in data.columns:
    encoder = LabelEncoder()
    data["Sex"] = encoder.fit_transform(data["Sex"])

if "Age" in data.columns:
    scaler = StandardScaler()
    data["Age"] = scaler.fit_transform(data[["Age"]])

print("\n\n\n***Processed Data:\n")
print(data.head())

data.to_csv("Titanic_Cleaned.csv", index = False)

print("\n\n\nDataset Cleaned Successfully!")
print("Cleaned File Saved as Titanic_Cleaned.csv")
