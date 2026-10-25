
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

data=load_breast_cancer()
X= data.data
y= data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dt= DecisionTreeClassifier(random_state=42)
rf= RandomForestClassifier(n_estimators=100, random_state=42)
ab= AdaBoostClassifier(n_estimators=50, random_state=42)

models = [dt,rf,ab]
names = ["Decision Tree","Random Forest", "AdaBoost"]
accuracy = []

for model in models:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = accuracy_score(y_test, pred)
    accuracy.append(score)
    print(type(model).__name__, "Accuracy:", round(score * 100, 2), "%")

plt.bar(names, accuracy)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

rf_pred = rf.predict(X_test)
ConfusionMatrixDisplay.from_predictions(
    y_test, rf_pred, display_labels=data.target_names
)
plt.title("Random Forest Confusion Matrix")
plt.show()

plt.barh(data.feature_names[:10], rf.feature_importances_[:10])
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.show()

best = accuracy.index(max(accuracy))
print("Best Model:", names[best])
print("Best Accuracy:", round(accuracy[best] * 100, 2), "%")
