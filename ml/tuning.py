
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
data = pd.read_csv(
    r"C:\Users\kavin\Downloads\CancerRisk\data\processed\cleaned_cervical_cancer.csv"
)

# 2. Separate features and target
X = data.drop(columns=["Biopsy"])
y = data["Biopsy"].astype(int)

print("Target distribution:")
print(y.value_counts())

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Create a pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    ))
])

# 5. Define hyperparameter combinations
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 5, 10, 15, 20],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}

# 6. Set up randomized search
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=20,
    scoring="recall",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# 7. Find the best hyperparameters
print("\nSearching for the best hyperparameters...")
search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\n========== TUNING RESULTS ==========")
print("Best Parameters:")
print(search.best_params_)
print("\nBest Cross-Validation Recall:")
print(round(search.best_score_, 4))

# 8. Evaluate the tuned model on the untouched test set
y_pred = best_model.predict(X_test)

print("\n========== TUNED MODEL EVALUATION ==========")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(
    precision_score(y_test, y_pred, zero_division=0), 4
))
print("Recall:", round(
    recall_score(y_test, y_pred, zero_division=0), 4
))
print("F1-Score:", round(
    f1_score(y_test, y_pred, zero_division=0), 4
))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Negative", "Positive"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nHyperparameter tuning completed!")