"""Transparent portfolio summaries and hypothetical loss-scenario calculations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def calculate_expected_loss(
    event_probability: float, loss_given_default: float, exposure_at_default: float
) -> float:
    """Calculate hypothetical EL = event probability * LGD * EAD."""
    probability = _bounded_fraction("Event probability", event_probability)
    lgd = _bounded_fraction("LGD assumption", loss_given_default)
    try:
        ead = float(exposure_at_default)
    except (TypeError, ValueError) as error:
        raise ValueError("EAD assumption must be a finite, non-negative amount.") from error
    if not np.isfinite(ead) or ead < 0:
        raise ValueError("EAD assumption must be a finite, non-negative amount.")
    return probability * lgd * ead


def _bounded_fraction(name: str, value: float) -> float:
    try:
        fraction = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite fraction between 0 and 1.") from error
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError(f"{name} must be a finite fraction between 0 and 1.")
    return fraction


def summarize_default_rates(
    data: pd.DataFrame,
    feature: str,
    target: str = "loan_status",
    minimum_group_size: int = 1,
) -> pd.DataFrame:
    """Return counts and observed label rates, preserving missing categories."""
    if feature not in data:
        raise ValueError(f"Portfolio field is unavailable: {feature}")
    if target not in data:
        raise ValueError(f"Target field is unavailable: {target}")
    if isinstance(minimum_group_size, bool) or not isinstance(minimum_group_size, int) or minimum_group_size < 1:
        raise ValueError("minimum_group_size must be a positive integer.")
    if data[target].isna().any() or not data[target].isin([0, 1]).all():
        raise ValueError("Portfolio target must contain complete binary labels.")

    result = (
        data.groupby(feature, dropna=False, observed=True)[target]
        .agg(applications="size", observed_defaults="sum", observed_default_rate="mean")
        .reset_index()
    )
    result = result.loc[result["applications"] >= minimum_group_size].copy()
    result["observed_default_rate_percent"] = 100 * result["observed_default_rate"]
    result["category"] = result[feature].astype("string").fillna("Missing / not recorded")
    return result.sort_values("observed_default_rate", ascending=False).reset_index(drop=True)


def get_global_feature_importance(artifact: Mapping[str, Any]) -> pd.DataFrame:
    """Average fitted tree importances across calibration folds and group encoded fields."""
    model_pipeline = artifact["model"]
    calibrated_models = getattr(model_pipeline, "calibrated_classifiers_", None)
    if not calibrated_models:
        raise ValueError("The saved calibrated pipeline does not expose fitted calibration estimators.")

    feature_columns = list(artifact["features"])
    fold_importances: list[pd.Series] = []
    for calibrated_model in calibrated_models:
        estimator = getattr(calibrated_model, "estimator", None)
        if estimator is None:
            estimator = getattr(calibrated_model, "estimator_", None)
        if estimator is None or not hasattr(estimator, "named_steps"):
            raise ValueError("A calibrated estimator does not expose its fitted pipeline.")
        preprocessor = estimator.named_steps.get("preprocessor")
        fitted_model = estimator.named_steps.get("classifier")
        if fitted_model is None:
            fitted_model = estimator.named_steps.get("model")
        if preprocessor is None or not hasattr(fitted_model, "feature_importances_"):
            raise ValueError("The fitted model does not expose tree-based feature importances.")

        transformed_names = preprocessor.get_feature_names_out()
        importances = np.asarray(fitted_model.feature_importances_, dtype=float)
        if importances.ndim != 1 or len(importances) != len(transformed_names):
            raise ValueError("Transformed feature names and fitted importance values do not align.")

        grouped_importance: dict[str, float] = {}
        for transformed_name, importance in zip(transformed_names, importances):
            local_name = transformed_name.split("__", 1)[-1]
            feature_name = next(
                (
                    column
                    for column in sorted(feature_columns, key=len, reverse=True)
                    if local_name == column or local_name.startswith(f"{column}_")
                ),
                local_name,
            )
            grouped_importance[feature_name] = grouped_importance.get(feature_name, 0.0) + float(importance)
        fold_importances.append(pd.Series(grouped_importance, dtype=float))

    result = pd.concat(fold_importances, axis=1).fillna(0.0)
    return (
        result.mean(axis=1)
        .rename("mean_feature_importance")
        .sort_values(ascending=False)
        .rename_axis("feature")
        .reset_index()
    )
