"""
Trains the loan eligibility model following the same pipeline as the
uploaded notebook (log transforms, label encoding, RandomForest with
tuned hyperparameters) and saves a portable artifact for the API.
"""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

df = pd.read_csv("Loan_Eligibility_Prediction.csv")

# Feature engineering (mirrors the notebook)
df["Total_Income"] = df["Applicant_Income"] + df["Coapplicant_Income"]
df["ApplicantIncomeLog"] = np.log(df["Applicant_Income"].replace(0, 1))
df["LoanAmountLog"] = np.log(df["Loan_Amount"] + 1)
df["Loan_Amount_Term_Log"] = np.log(df["Loan_Amount_Term"] + 1)
df["Total_Income_Log"] = np.log(df["Total_Income"] + 1)

cat_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = list(le.classes_)  # index -> original label

feature_cols = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "Credit_History", "Property_Area",
    "ApplicantIncomeLog", "LoanAmountLog", "Loan_Amount_Term_Log", "Total_Income_Log",
]

X = df[feature_cols]
y = df["Loan_Status"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = RandomForestClassifier(
    n_estimators=100, min_samples_split=25, max_depth=7, max_features=1, random_state=42
)
model.fit(x_train, y_train)

acc = accuracy_score(y_test, model.predict(x_test)) * 100
cv = cross_val_score(model, X, y, cv=5).mean() * 100
cm = confusion_matrix(y_test, model.predict(x_test)).tolist()

print("Test accuracy:", acc)
print("Cross-val accuracy:", cv)
print("Confusion matrix:", cm)

joblib.dump(model, "model.joblib")

metadata = {
    "feature_cols": feature_cols,
    "encoders": encoders,  # e.g. Gender: ["Female", "Male"] -> 0,1
    "metrics": {"test_accuracy": acc, "cv_accuracy": cv, "confusion_matrix": cm},
    "feature_importance": dict(zip(feature_cols, model.feature_importances_.tolist())),
}
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Saved model.joblib and model_metadata.json")
