
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# 1. Load the same dataset used for tuning
data = pd.read_csv(
    r"C:\Users\kavin\Downloads\CancerRisk\data\processed\cleaned_cervical_cancer.csv"
)

X = data.drop(columns=["Biopsy"])
y = data["Biopsy"].astype(int)

# 2. Create one common train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Original Random Forest (before tuning)
before_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2
    ))
])

# 4. Tuned Random Forest (best parameters from tuning.py)
after_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=4,
        random_state=42,
        class_weight="balanced"
    ))
])

# 5. Train both models on the same training data
before_model.fit(X_train, y_train)
after_model.fit(X_train, y_train)

# 6. Evaluate both models on the same test data
before_pred = before_model.predict(X_test)
after_pred = after_model.predict(X_test)

def get_metrics(y_true, predictions):
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1-Score": f1_score(
            y_true, predictions, zero_division=0
        )
    }

before_results = get_metrics(y_test, before_pred)
after_results = get_metrics(y_test, after_pred)

# 7. Display comparison
comparison = pd.DataFrame(
    [before_results, after_results],
    index=["Before Tuning", "After Tuning"]
)

print("\n========== BEFORE vs AFTER TUNING ==========")
print(comparison.round(4))

print("\n========== BEFORE TUNING CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, before_pred, labels=[0, 1]))

print("\n========== AFTER TUNING CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, after_pred, labels=[0, 1]))

print("\n========== COMPARISON SUMMARY ==========")

for metric in before_results:
    difference = after_results[metric] - before_results[metric]
    print(f"{metric} change: {difference:+.4f}")

print("\nModel comparison completed!")