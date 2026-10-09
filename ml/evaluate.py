import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,precision_score,recall_score,f1_score

data = pd.read_csv(r"C:\Users\kavin\Downloads\CancerRisk\data\processed\cleaned_cervical_cancer.csv")

X = data.drop(columns=["Biopsy"])
Y = data["Biopsy"].astype(int)

print("Target distribution:")
print(data["Biopsy"].value_counts())

print("\nTarget percentages:")
print(data["Biopsy"].value_counts(normalize=True) * 100)

X_train,X_test,Y_train,Y_test = train_test_split(
    X,Y,test_size=0.2,random_state=42,stratify=Y)

model = RandomForestClassifier(n_estimators=100,random_state=42,class_weight="balanced",
    min_samples_leaf=2)

model.fit(X_train,Y_train)

y_pred = model.predict(X_test)

print("========== MODEL EVALUATION ==========")

print("Accuracy:", accuracy_score(Y_test, y_pred))
print("Precision:", precision_score(Y_test, y_pred, zero_division=0))
print("Recall:", recall_score(Y_test, y_pred, zero_division=0))
print("F1-Score:", f1_score(Y_test, y_pred, zero_division=0))

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(Y_test,y_pred,labels=[0, 1],target_names=["Negative", "Positive"],zero_division=0))


print("\n========== CONFUSION MATRIX ==========")
cm = confusion_matrix(Y_test, y_pred, labels=[0, 1])
print(cm)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.show()

print("\nEvaluation completed successfully!")

