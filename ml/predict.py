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

#New feature data
print("\nEnter the feature values for one new patient:")

patient_data = {}

for feature in X.columns:
    value = float(input(f"{feature}: "))
    patient_data[feature] = value

# Convert input into a DataFrame
new_patient = pd.DataFrame([patient_data], columns=X.columns)

# Predict
prediction = model.predict(new_patient)[0]
probabilities = model.predict_proba(new_patient)[0]

# Display the result
print("\n========== PREDICTION RESULT ==========")

if prediction == 1:
    print("Prediction: POSITIVE (Biopsy = 1)")
else:
    print("Prediction: NEGATIVE (Biopsy = 0)")

print(f"Probability of Negative: {probabilities[0] * 100:.2f}%")
print(f"Probability of Positive: {probabilities[1] * 100:.2f}%")

print("\nPrediction completed!")