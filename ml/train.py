import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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

print("Actual:",Y_test)
print("Predicted:",y_pred)

# 1. Display training class distribution
print("\nTraining Class Distribution:")
print(Y_train.value_counts().sort_index())

# 2. Calculate training accuracy
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, y_train_pred)

print("\nTraining Accuracy:",
      round(train_accuracy * 100, 2), "%")

# 3. Display top 10 important features
rf_model = model

importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Important Features:")
print(importance.head(10))

print("Model trained successfully")