# Import Libraries

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset

data = load_breast_cancer()

X = data.data
y = data.target

# Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Standardize the Features

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create Logistic Regression Model

model = LogisticRegression()

# Train the Model

model.fit(X_train, y_train)

# Make Predictions

y_pred = model.predict(X_test)

# Calculate Accuracy

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy =", accuracy)

# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Precision

precision = precision_score(y_test, y_pred)

print("\nPrecision =", precision)

# Recall

recall = recall_score(y_test, y_pred)

print("\nRecall =", recall)

# F1 Score

f1 = f1_score(y_test, y_pred)

print("\nF1 Score =", f1)

# Classification Report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display Confusion Matrix

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Malignant", "Benign"],
    yticklabels=["Malignant", "Benign"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")

plt.show()
