"""Safe, stateless inference helpers for the trusted, fitted credit-risk pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "credit_risk_model.joblib"
DERIVED_SOURCE_FIELDS = {"estimated_emi_inr", "debt_to_income_after_loan"}


def calculate_monthly_emi(
    principal_inr: float, annual_interest_rate_pct: float, tenure_months: int
) -> float:
    """Calculate a standard fixed-rate monthly EMI for a positive principal."""
    principal = _finite_number("requested_loan_amount_inr", principal_inr)
    annual_rate = _finite_number("annual_interest_rate_pct", annual_interest_rate_pct)
    months = _integer("loan_tenure_months", tenure_months)
    if principal <= 0:
        raise ValueError("Requested loan amount must be greater than zero.")
    if annual_rate < 0 or annual_rate > 100:
        raise ValueError("Annual interest rate must be between 0 and 100 percent.")
    if months <= 0:
        raise ValueError("Loan tenure must be a positive number of months.")

    monthly_rate = annual_rate / 1200
    if monthly_rate == 0:
        return principal / months
    growth_factor = (1 + monthly_rate) ** months
    return principal * monthly_rate * growth_factor / (growth_factor - 1)


def load_model_artifact(model_path: str | Path | None = None) -> dict[str, Any]:
    """Load a trusted Joblib artifact and validate its inference contract.

    Never load a model artifact from an untrusted source: deserializing a Joblib
    file can execute code.
    """
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Fitted model artifact was not found: {path}")

    import joblib

    artifact = joblib.load(path)
    required_keys = {
        "pipeline",
        "target_column",
        "positive_label",
        "decision_threshold",
        "raw_feature_columns",
        "model_input_columns",
        "engineered_features",
        "test_metrics",
    }
    if not isinstance(artifact, dict):
        raise ValueError(f"Model artifact must be a dictionary: {path}")
    missing_keys = required_keys.difference(artifact)
    if missing_keys:
        raise ValueError(f"Model artifact is missing required metadata: {sorted(missing_keys)}")
    if not callable(getattr(artifact["pipeline"], "predict_proba", None)):
        raise ValueError("The saved estimator does not implement predict_proba.")
    if not 0 <= float(artifact["decision_threshold"]) <= 1:
        raise ValueError("Saved classification threshold must be between 0 and 1.")
    if set(artifact["engineered_features"]) != {"requested_loan_to_annual_income"}:
        raise ValueError("Unexpected or unsupported engineered feature in model metadata.")
    return artifact


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number.") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


def _integer(name: str, value: Any) -> int:
    numeric = _finite_number(name, value)
    if not numeric.is_integer():
        raise ValueError(f"{name} must be a whole number.")
    return int(numeric)


def _validate_application(inputs: Mapping[str, Any], required: set[str]) -> dict[str, Any]:
    if not isinstance(inputs, Mapping):
        raise ValueError("Borrower inputs must be supplied as a mapping of field names to values.")
    missing = required.difference(inputs)
    unexpected = set(inputs).difference(required)
    if missing:
        raise ValueError(f"Required application fields are missing: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"Unexpected application fields were supplied: {sorted(unexpected)}")

    values = dict(inputs)
    integer_fields = {
        "age_years",
        "loan_tenure_months",
        "open_credit_accounts",
        "credit_inquiries_last_6m",
        "late_payments_last_12m",
        "dependents_count",
    }
    numeric_fields = required.difference(
        {"state", "city_tier", "employment_type", "education_level", "residence_type", "loan_purpose"}
    )

    for field in numeric_fields:
        values[field] = _finite_number(field, values[field])
    for field in integer_fields:
        values[field] = _integer(field, values[field])

    if not 18 <= values["age_years"] <= 100:
        raise ValueError("Applicant age must be between 18 and 100 years.")
    if not 0 <= values["employment_years"] <= values["age_years"]:
        raise ValueError("Employment duration must be non-negative and no greater than applicant age.")
    if values["monthly_net_income_inr"] <= 0:
        raise ValueError("Monthly net income must be greater than zero.")
    if values["existing_monthly_debt_inr"] < 0:
        raise ValueError("Existing monthly debt cannot be negative.")
    if values["requested_loan_amount_inr"] <= 0:
        raise ValueError("Requested loan amount must be greater than zero.")
    if values["loan_tenure_months"] <= 0:
        raise ValueError("Loan tenure must be a positive number of months.")
    if not 0 <= values["annual_interest_rate_pct"] <= 100:
        raise ValueError("Annual interest rate must be between 0 and 100 percent.")
    if values["credit_history_years"] < 0 or values["credit_history_years"] > values["age_years"]:
        raise ValueError("Credit history must be non-negative and no greater than applicant age.")
    for field in ("open_credit_accounts", "credit_inquiries_last_6m", "late_payments_last_12m", "dependents_count"):
        if values[field] < 0:
            raise ValueError(f"{field} cannot be negative.")
    if values["credit_utilization_ratio"] < 0:
        raise ValueError("Credit utilization ratio cannot be negative.")
    if values["savings_balance_inr"] < 0:
        raise ValueError("Savings balance cannot be negative.")
    if values["banking_relationship_months"] < 0:
        raise ValueError("Banking relationship duration cannot be negative.")
    for field in ("state", "city_tier", "employment_type", "education_level", "residence_type", "loan_purpose"):
        if not isinstance(values[field], str) or not values[field].strip():
            raise ValueError(f"{field} must be a non-empty category.")
        values[field] = values[field].strip()
    return values


def prepare_model_input(
    inputs: Mapping[str, Any], artifact: Mapping[str, Any]
) -> pd.DataFrame:
    """Validate form inputs and build columns in the model's saved input order."""
    raw_columns = list(artifact["raw_feature_columns"])
    expected_user_fields = set(raw_columns).difference(DERIVED_SOURCE_FIELDS)
    values = _validate_application(inputs, expected_user_fields)

    values["estimated_emi_inr"] = calculate_monthly_emi(
        values["requested_loan_amount_inr"],
        values["annual_interest_rate_pct"],
        values["loan_tenure_months"],
    )
    values["debt_to_income_after_loan"] = (
        values["existing_monthly_debt_inr"] + values["estimated_emi_inr"]
    ) / values["monthly_net_income_inr"]
    raw_row = pd.DataFrame([{column: values[column] for column in raw_columns}])
    raw_row["requested_loan_to_annual_income"] = (
        raw_row["requested_loan_amount_inr"] / (12 * raw_row["monthly_net_income_inr"])
    )

    model_columns = list(artifact["model_input_columns"])
    if set(model_columns) != set(raw_row.columns):
        raise ValueError("Saved model input columns do not match the inference feature contract.")
    model_row = raw_row.loc[:, model_columns]
    if not np.isfinite(model_row.select_dtypes(include="number").to_numpy(dtype=float)).all():
        raise ValueError("Validated application produced non-finite numeric model inputs.")
    return model_row


def predict_application(
    inputs: Mapping[str, Any], artifact: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a descriptive event probability and threshold-based demo label."""
    model_input = prepare_model_input(inputs, artifact)
    probabilities = np.asarray(artifact["pipeline"].predict_proba(model_input), dtype=float)
    classes = list(artifact["pipeline"].classes_)
    positive_label = artifact["positive_label"]
    if probabilities.ndim != 2 or probabilities.shape != (1, len(classes)):
        raise ValueError(f"Unexpected prediction-probability shape: {probabilities.shape}")
    if positive_label not in classes:
        raise ValueError("The positive event label is absent from the fitted estimator.")
    event_probability = float(probabilities[0, classes.index(positive_label)])
    if not np.isfinite(event_probability) or not 0 <= event_probability <= 1:
        raise ValueError("The fitted model returned an invalid event probability.")

    threshold = float(artifact["decision_threshold"])
    return {
        "event_probability": event_probability,
        "decision_threshold": threshold,
        "predicted_label": int(event_probability >= threshold),
        "model_name": artifact.get("model_name", "Fitted classifier"),
        "event_definition": f"Observed {artifact['target_column']} label equals {positive_label}",
        "model_input": model_input,
    }
