
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ==================================================
# 1. LOAD DATASET
# ==================================================

data = pd.read_csv(
    r"C:\Users\kavin\Downloads\CancerRisk\data\raw\cervical_cancer.csv"
)

print("Dataset Shape:", data.shape)
print("\nFirst 5 Rows:")
print(data.head())

print("\nDataset Information:")
data.info()

# ==================================================
# 2. DATA PREPROCESSING
# ==================================================

# Replace missing-value symbols with NaN
data.replace("?", np.nan, inplace=True)

# Convert all columns to numeric
data = data.apply(pd.to_numeric, errors="coerce")

# Remove duplicate rows
data.drop_duplicates(inplace=True)

# Remove rows with missing target
target = "Biopsy"
data.dropna(subset=[target], inplace=True)

# Ensure target contains only valid binary values
data = data[data[target].isin([0, 1])].copy()
data[target] = data[target].astype(int)

# Remove other diagnostic outcomes to reduce data leakage
remove_cols = [
    "Hinselmann",
    "Schiller",
    "Citology",
    "Dx:Cancer",
    "Dx:CIN",
    "Dx:HPV",
    "Dx"
]

X = data.drop(
    columns=[target] + remove_cols,
    errors="ignore"
)

y = data[target]

# Remove columns containing only missing values
X.dropna(axis=1, how="all", inplace=True)

# Remove constant features
constant_cols = [
    col for col in X.columns
    if X[col].nunique(dropna=True) <= 1
]

X.drop(columns=constant_cols, inplace=True)

print("\nCleaned Dataset Shape:", X.shape)
print("Target Distribution:")
print(y.value_counts())

# ==================================================
# 3. VISUALIZE TARGET DISTRIBUTION
# ==================================================

sns.countplot(x=y)
plt.title("Biopsy Outcome Distribution")
plt.xlabel("Biopsy (0 = Negative, 1 = Positive)")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()

# ==================================================
# 4. VISUALIZE MISSING VALUES
# ==================================================

missing = X.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if not missing.empty:
    missing.plot(kind="bar", figsize=(12, 5))
    plt.title("Missing Values by Feature")
    plt.ylabel("Missing Count")
    plt.tight_layout()
    plt.show()
else:
    print("No missing feature values.")

# ==================================================
# 5. CORRELATION HEATMAP
# ==================================================

# This imputation is only for exploratory visualization.
# Model training will use a separate training-only imputer.

if X.shape[1] > 1:
    viz_imputer = SimpleImputer(strategy="median")

    X_viz = pd.DataFrame(
        viz_imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    plt.figure(figsize=(15, 10))
    sns.heatmap(
        X_viz.corr(),
        cmap="coolwarm",
        center=0
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

# ==================================================
# 6. SPLIT DATASET
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ==================================================
# 7. HANDLE MISSING VALUES WITHOUT DATA LEAKAGE
# ==================================================

imputer = SimpleImputer(strategy="median")

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X.columns,
    index=X_train.index
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X.columns,
    index=X_test.index
)

# ==================================================
# 8. FEATURE IMPORTANCE
# ==================================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

importance = pd.Series(
    rf.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(10, 8))
importance.sort_values().plot(kind="barh")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# ==================================================
# 9. SELECT FEATURES FOR FURTHER EXPERIMENTS
# ==================================================

# Keep features with importance above zero.
# This is an initial selection, not proof that other
# features are unnecessary.

selected_features = importance[importance > 0].index.tolist()

print("\nOriginal Feature Count:", X_train.shape[1])
print("Selected Feature Count:", len(selected_features))
print("\nSelected Features:")
print(selected_features)

# Create reduced datasets for comparison
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# ==================================================
# 10. SAVE CLEANED DATASET
# ==================================================

cleaned_data = X.copy()
cleaned_data[target] = y

cleaned_data.to_csv(
    "cleaned_cervical_cancer.csv",
    index=False
)

print("\nPreprocessing completed successfully!")
print("Cleaned dataset saved as cleaned_cervical_cancer.csv")
print("Training and testing data are ready for model training.")