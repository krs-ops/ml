import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset shape:", X.shape)
print("Number of Features:", X.shape[1])
print("Number of Classes:", len(np.unique(y)))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Linear SVM
linear_svm = SVC(kernel='linear')
linear_svm.fit(X_train, y_train)

y_pred_linear = linear_svm.predict(X_test)


# RBF SVM
rbf_svm = SVC(kernel='rbf')
rbf_svm.fit(X_train, y_train)

y_pred_rbf = rbf_svm.predict(X_test)


def evaluate_model(name, y_test, y_pred):

    print("\n", name)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return accuracy, precision, recall, f1


linear_results = evaluate_model(
    "Linear SVM",
    y_test,
    y_pred_linear
)

rbf_results = evaluate_model(
    "Nonlinear SVM (RBF Kernel)",
    y_test,
    y_pred_rbf
)


print("\n\n== MODEL COMPARISON ==")

print("\nMetric\t\tLinear SVM\tRBF SVM")

print(
    "Accuracy\t",
    round(linear_results[0], 4),
    "\t\t",
    round(rbf_results[0], 4)
)

print(
    "Precision\t",
    round(linear_results[1], 4),
    "\t\t",
    round(rbf_results[1], 4)
)

print(
    "Recall\t\t",
    round(linear_results[2], 4),
    "\t\t",
    round(rbf_results[2], 4)
)

print(
    "F1 Score\t",
    round(linear_results[3], 4),
    "\t\t",
    round(rbf_results[3], 4)
)


metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

linear_values = [
    linear_results[0],
    linear_results[1],
    linear_results[2],
    linear_results[3]
]

rbf_values = [
    rbf_results[0],
    rbf_results[1],
    rbf_results[2],
    rbf_results[3]
]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(9, 6))

plt.bar(
    x - width / 2,
    linear_values,
    width,
    label="Linear SVM"
)

plt.bar(
    x + width / 2,
    rbf_values,
    width,
    label="RBF SVM"
)

plt.xlabel("Evaluation Metrics")
plt.ylabel("Score")
plt.title("Linear SVM vs Nonlinear SVM")

plt.xticks(x, metrics)
plt.ylim(0, 1.1)

plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.show()
