
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

# 1. Load dataset
data = pd.read_csv(
    r"C:\Users\kavin\Downloads\CancerRisk\data\processed\cleaned_cervical_cancer.csv"
)

# 2. Separate features and target
X = data.drop(columns=["Biopsy"])
Y = data["Biopsy"].astype(int)

# 3. Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

# 4. Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    min_samples_leaf=2
)

model.fit(X_train, Y_train)

# 5. Select the first decision tree
first_tree = model.estimators_[0]

print("Random Forest trained successfully")
print("Total trees:", len(model.estimators_))
print("First tree depth:", first_tree.get_depth())
print("First tree leaf nodes:", first_tree.get_n_leaves())

# 6. Visualize the first tree
plt.figure(figsize=(24, 12))

plot_tree(
    first_tree,
    feature_names=X.columns,
    class_names=["Negative", "Positive"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=9
)

plt.title("First Decision Tree in Random Forest")
plt.tight_layout()
plt.show()