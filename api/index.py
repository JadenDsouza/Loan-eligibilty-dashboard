"""
FastAPI backend for the Loan Eligibility dashboard.
Vercel routes all /api/* requests into this file (see vercel.json).
"""
import json
import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    LoanApplication,
    PredictionResponse,
    ModelMetrics,
    FeatureImportance,
    DatasetSummary,
    HealthResponse,
    DashboardData,
    GenderSplit,
    DependentBar,
    PropertyIncomeBar,
    TermApprovalPoint,
    IncomeApprovalPoint,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="Loan Eligibility API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_model():
    path = os.path.join(BASE_DIR, "model.joblib")
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_metadata():
    path = os.path.join(BASE_DIR, "model_metadata.json")
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_dataset():
    path = os.path.join(BASE_DIR, "Loan_Eligibility_Prediction.csv")
    return pd.read_csv(path)


def encode_value(field: str, value: str, encoders: dict) -> int:
    classes = encoders[field]
    try:
        return classes.index(value)
    except ValueError:
        raise HTTPException(400, f"Unknown value '{value}' for {field}")


@app.get("/api/health", response_model=HealthResponse)
def health():
    try:
        get_model()
        loaded = True
    except Exception:
        loaded = False
    return HealthResponse(status="ok", model_loaded=loaded)


@app.post("/api/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    model = get_model()
    meta = get_metadata()
    encoders = meta["encoders"]

    gender = encode_value("Gender", application.gender.value, encoders)
    married = encode_value("Married", application.married.value, encoders)
    education = encode_value("Education", application.education.value, encoders)
    self_employed = encode_value("Self_Employed", application.self_employed.value, encoders)
    property_area = encode_value("Property_Area", application.property_area.value, encoders)

    total_income = application.applicant_income + application.coapplicant_income
    applicant_income_log = np.log(max(application.applicant_income, 1))
    loan_amount_log = np.log(application.loan_amount + 1)
    loan_term_log = np.log(application.loan_amount_term + 1)
    total_income_log = np.log(total_income + 1)

    row = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": application.dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "Credit_History": application.credit_history,
        "Property_Area": property_area,
        "ApplicantIncomeLog": applicant_income_log,
        "LoanAmountLog": loan_amount_log,
        "Loan_Amount_Term_Log": loan_term_log,
        "Total_Income_Log": total_income_log,
    }])[meta["feature_cols"]]

    proba = model.predict_proba(row)[0]
    pred = model.predict(row)[0]

    loan_status_labels = encoders["Loan_Status"]  # index -> "N"/"Y"
    status_label = loan_status_labels[pred]
    y_index = loan_status_labels.index("Y") if "Y" in loan_status_labels else 1
    n_index = 1 - y_index

    return PredictionResponse(
        loan_status=status_label,
        approved=(status_label == "Y"),
        approval_probability=float(proba[y_index]),
        rejection_probability=float(proba[n_index]),
    )


@app.get("/api/metrics", response_model=ModelMetrics)
def metrics():
    meta = get_metadata()
    m = meta["metrics"]
    importances = [
        FeatureImportance(feature=k, importance=v)
        for k, v in sorted(meta["feature_importance"].items(), key=lambda x: -x[1])
    ]
    return ModelMetrics(
        test_accuracy=m["test_accuracy"],
        cv_accuracy=m["cv_accuracy"],
        confusion_matrix=m["confusion_matrix"],
        feature_importance=importances,
    )


@app.get("/api/dataset-summary", response_model=DatasetSummary)
def dataset_summary():
    df = get_dataset()
    approved = int((df["Loan_Status"] == "Y").sum())
    rejected = int((df["Loan_Status"] == "N").sum())
    total = len(df)

    return DatasetSummary(
        total_records=total,
        approved_count=approved,
        rejected_count=rejected,
        approval_rate=round(approved / total * 100, 2),
        avg_applicant_income=round(df["Applicant_Income"].mean(), 2),
        avg_loan_amount=round(df["Loan_Amount"].mean(), 2),
        by_property_area=df.groupby("Property_Area")["Loan_Status"]
        .apply(lambda s: round((s == "Y").mean() * 100, 1)).to_dict(),
        by_education=df.groupby("Education")["Loan_Status"]
        .apply(lambda s: round((s == "Y").mean() * 100, 1)).to_dict(),
        by_credit_history=df.groupby("Credit_History")["Loan_Status"]
        .apply(lambda s: round((s == "Y").mean() * 100, 1)).to_dict(),
    )


@app.get("/api/dashboard-data", response_model=DashboardData)
def dashboard_data():
    """Single aggregate payload powering every chart on the dashboard,
    mirroring the reference BI-report layout (KPIs, gender split,
    dependents vs loan amount, income by area, approvals by term)."""
    df = get_dataset()
    total = len(df)

    male_ct = int((df["Gender"] == "Male").sum())
    female_ct = int((df["Gender"] == "Female").sum())

    gender_split = GenderSplit(
        male=male_ct,
        female=female_ct,
        male_pct=round(male_ct / total * 100, 2),
        female_pct=round(female_ct / total * 100, 2),
    )

    dep_avg = df.groupby("Dependents")["Loan_Amount"].mean().sort_index()
    avg_loan_by_dependents = [
        DependentBar(dependents=str(k), avg_loan_amount=round(v, 1))
        for k, v in dep_avg.items()
    ]

    area_income = df.groupby("Property_Area")["Applicant_Income"].sum().sort_values(ascending=False)
    income_by_property_area = [
        PropertyIncomeBar(property_area=k, total_applicant_income=float(v))
        for k, v in area_income.items()
    ]

    bins = [0, 100, 200, 300, 400, 500]
    labels = ["0-100", "100-200", "200-300", "300-400", "400-500"]
    df["_term_bucket"] = pd.cut(df["Loan_Amount_Term"], bins=bins, labels=labels)
    term_approval = (
        df.groupby("_term_bucket", observed=True)["Loan_Status"]
        .apply(lambda s: round((s == "Y").mean() * 100, 1) if len(s) else 0.0)
    )
    approval_by_loan_term = [
        TermApprovalPoint(term_bucket=str(k), approval_pct=float(v))
        for k, v in term_approval.items()
    ]

    income_bins = sorted(df["Applicant_Income"].unique())
    sample = df.sort_values("Applicant_Income")
    approval_by_income = [
        IncomeApprovalPoint(income=float(row.Applicant_Income), approval_pct=100.0 if row.Loan_Status == "Y" else 0.0)
        for row in sample.itertuples()
    ][:120]  # cap payload size; frontend renders a scatter/line trend

    return DashboardData(
        total_records=total,
        self_employed_count=int((df["Self_Employed"] == "Yes").sum()),
        graduate_count=int((df["Education"] == "Graduate").sum()),
        approval_rate=round((df["Loan_Status"] == "Y").mean() * 100, 2),
        credit_history_approval_rate=round(
            df[df["Credit_History"] == 1]["Loan_Status"].eq("Y").mean() * 100, 2
        ),
        avg_applicant_income=round(df["Applicant_Income"].mean(), 1),
        avg_coapplicant_income=round(df["Coapplicant_Income"].mean(), 1),
        avg_loan_amount=round(df["Loan_Amount"].mean(), 2),
        gender_split=gender_split,
        avg_loan_by_dependents=avg_loan_by_dependents,
        income_by_property_area=income_by_property_area,
        approval_by_loan_term=approval_by_loan_term,
        approval_by_income=approval_by_income,
    )
