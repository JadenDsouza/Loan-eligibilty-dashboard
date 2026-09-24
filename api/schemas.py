"""
Pydantic models for the Loan Eligibility API.
These define the exact shape and validation rules for every request
and response the dashboard talks to.
"""
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class GenderEnum(str, Enum):
    male = "Male"
    female = "Female"


class YesNoEnum(str, Enum):
    yes = "Yes"
    no = "No"


class EducationEnum(str, Enum):
    graduate = "Graduate"
    not_graduate = "Not Graduate"


class PropertyAreaEnum(str, Enum):
    urban = "Urban"
    semiurban = "Semiurban"
    rural = "Rural"


class LoanApplication(BaseModel):
    """A single loan applicant's data, validated before scoring."""

    gender: GenderEnum
    married: YesNoEnum
    dependents: int = Field(ge=0, le=10, description="Number of dependents")
    education: EducationEnum
    self_employed: YesNoEnum
    applicant_income: float = Field(gt=0, description="Monthly applicant income")
    coapplicant_income: float = Field(ge=0, description="Monthly co-applicant income")
    loan_amount: float = Field(gt=0, description="Requested loan amount (in thousands)")
    loan_amount_term: float = Field(gt=0, description="Loan term in days/months")
    credit_history: int = Field(description="1 = good credit history, 0 = bad")
    property_area: PropertyAreaEnum

    @field_validator("credit_history")
    @classmethod
    def validate_credit_history(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("credit_history must be 0 or 1")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Male",
                "married": "Yes",
                "dependents": 1,
                "education": "Graduate",
                "self_employed": "No",
                "applicant_income": 5000,
                "coapplicant_income": 1500,
                "loan_amount": 120,
                "loan_amount_term": 360,
                "credit_history": 1,
                "property_area": "Urban",
            }
        }
    }


class PredictionResponse(BaseModel):
    """Model output returned to the dashboard."""

    loan_status: str = Field(description="'Y' (approved) or 'N' (rejected)")
    approved: bool
    approval_probability: float = Field(ge=0, le=1)
    rejection_probability: float = Field(ge=0, le=1)


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class ModelMetrics(BaseModel):
    test_accuracy: float
    cv_accuracy: float
    confusion_matrix: List[List[int]]
    feature_importance: List[FeatureImportance]


class DatasetSummary(BaseModel):
    """Aggregate stats used to populate dashboard charts."""

    total_records: int
    approved_count: int
    rejected_count: int
    approval_rate: float
    avg_applicant_income: float
    avg_loan_amount: float
    by_property_area: dict
    by_education: dict
    by_credit_history: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
