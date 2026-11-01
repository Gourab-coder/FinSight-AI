"""Inference helpers for the trusted, fitted credit-risk classifier."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "credit_risk_model.joblib"


def _fitted_pipeline(model: Any) -> Any:
    if hasattr(model, "named_steps"):
        return model
    calibrated_models = getattr(model, "calibrated_classifiers_", None)
    if calibrated_models:
        estimator = getattr(calibrated_models[0], "estimator", None)
        if estimator is None:
            estimator = getattr(calibrated_models[0], "estimator_", None)
        if estimator is not None and hasattr(estimator, "named_steps"):
            return estimator
    raise ValueError("The saved classifier does not expose a fitted preprocessing pipeline.")


def get_model_input_schema(artifact: Mapping[str, Any]) -> dict[str, list[str]]:
    """Return the numeric and categorical input columns used by the fitted pipeline."""
    features = list(artifact["features"])
    preprocessor = _fitted_pipeline(artifact["model"]).named_steps.get("preprocessor")
    if preprocessor is None or not hasattr(preprocessor, "transformers_"):
        raise ValueError("The fitted pipeline does not expose its input preprocessing schema.")

    schema = {"numeric": [], "categorical": []}
    for name, _, columns in preprocessor.transformers_:
        schema_name = {"num": "numeric", "cat": "categorical"}.get(name)
        if schema_name is None:
            continue
        if isinstance(columns, str):
            selected_columns = [columns]
        else:
            selected_columns = list(columns)
        schema[schema_name].extend(selected_columns)

    all_columns = schema["numeric"] + schema["categorical"]
    if len(all_columns) != len(set(all_columns)) or set(all_columns) != set(features):
        raise ValueError("Fitted preprocessing columns do not match the saved feature list.")
    return schema


def load_model_artifact(model_path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate a trusted Joblib model artifact.

    Never load a model artifact from an untrusted source: deserializing a Joblib
    file can execute code.
    """
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Fitted model artifact was not found: {path}")

    import joblib

    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise ValueError(f"Model artifact must be a dictionary: {path}")
    missing_keys = {"model", "threshold", "risk_band_cutoffs", "features"}.difference(artifact)
    if missing_keys:
        raise ValueError(f"Model artifact is missing required metadata: {sorted(missing_keys)}")
    if not callable(getattr(artifact["model"], "predict_proba", None)):
        raise ValueError("The saved estimator does not implement predict_proba.")

    threshold = float(artifact["threshold"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Saved classification threshold must be between 0 and 1.")
    features = artifact["features"]
    if (
        not isinstance(features, (list, tuple))
        or not features
        or any(not isinstance(feature, str) or not feature for feature in features)
        or len(features) != len(set(features))
    ):
        raise ValueError("Saved model features must be a non-empty list of unique column names.")

    cutoffs = artifact["risk_band_cutoffs"]
    if not isinstance(cutoffs, Mapping) or not {"low_max", "medium_max"}.issubset(cutoffs):
        raise ValueError("Saved risk-band metadata must include low_max and medium_max.")
    low_max = float(cutoffs["low_max"])
    medium_max = float(cutoffs["medium_max"])
    if not (np.isfinite(low_max) and np.isfinite(medium_max) and 0 <= low_max < medium_max <= 1):
        raise ValueError("Saved risk-band cutoffs must satisfy 0 <= low_max < medium_max <= 1.")

    get_model_input_schema(artifact)
    return artifact


def _validated_inputs(inputs: Mapping[str, Any], artifact: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(inputs, Mapping):
        raise ValueError("Model inputs must be supplied as a mapping of feature names to values.")
    features = list(artifact["features"])
    missing = set(features).difference(inputs)
    unexpected = set(inputs).difference(features)
    if missing:
        raise ValueError(f"Required model inputs are missing: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"Unexpected model inputs were supplied: {sorted(unexpected)}")

    schema = get_model_input_schema(artifact)
    values = dict(inputs)
    for feature in schema["numeric"]:
        if isinstance(values[feature], bool):
            raise ValueError(f"{feature} must be a finite number.")
        try:
            number = float(values[feature])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{feature} must be a finite number.") from error
        if not np.isfinite(number):
            raise ValueError(f"{feature} must be a finite number.")
        values[feature] = number
    for feature in schema["categorical"]:
        if not isinstance(values[feature], str) or not values[feature].strip():
            raise ValueError(f"{feature} must be a non-empty category.")
        values[feature] = values[feature].strip()
    return values


def prepare_model_input(
    inputs: Mapping[str, Any], artifact: Mapping[str, Any]
) -> pd.DataFrame:
    """Validate one application and order its columns as required by the model."""
    values = _validated_inputs(inputs, artifact)
    return pd.DataFrame([{feature: values[feature] for feature in artifact["features"]}])


def predict_application(
    inputs: Mapping[str, Any], artifact: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the model's probability for dataset label 1 and threshold comparison."""
    model_input = prepare_model_input(inputs, artifact)
    model = artifact["model"]
    probabilities = np.asarray(model.predict_proba(model_input), dtype=float)
    classes = list(model.classes_)
    positive_label = 1
    if probabilities.ndim != 2 or probabilities.shape != (1, len(classes)):
        raise ValueError(f"Unexpected prediction-probability shape: {probabilities.shape}")
    if positive_label not in classes:
        raise ValueError("The positive event label 1 is absent from the fitted estimator.")
    event_probability = float(probabilities[0, classes.index(positive_label)])
    if not np.isfinite(event_probability) or not 0 <= event_probability <= 1:
        raise ValueError("The fitted model returned an invalid event probability.")

    threshold = float(artifact["threshold"])
    calibrated_models = getattr(model, "calibrated_classifiers_", [])
    estimator = getattr(calibrated_models[0], "estimator", None) if calibrated_models else None
    if estimator is None and calibrated_models:
        estimator = getattr(calibrated_models[0], "estimator_", None)
    classifier = estimator.named_steps.get("classifier") if hasattr(estimator, "named_steps") else None
    model_name = artifact.get(
        "model_name",
        f"Calibrated {type(classifier).__name__}" if classifier is not None else type(model).__name__,
    )
    return {
        "event_probability": event_probability,
        "decision_threshold": threshold,
        "predicted_label": int(event_probability >= threshold),
        "model_name": model_name,
        "event_definition": "loan_status = 1 (default label)",
        "model_input": model_input,
    }
