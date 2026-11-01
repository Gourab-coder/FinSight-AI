"""Streamlit dashboard for the educational FinSight AI credit-risk project."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.predict import (
    get_model_input_schema,
    load_model_artifact,
    predict_application,
)
from src.risk_analytics import (
    calculate_expected_loss,
    get_global_feature_importance,
    summarize_default_rates,
)

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "credit_risk_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "credit_risk_model.joblib"
TARGET = "loan_status"

st.set_page_config(page_title="FinSight AI | Credit Risk Lab", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner="Loading the trusted fitted model…")
def get_model_artifact():
    return load_model_artifact(MODEL_PATH)


@st.cache_data(show_spinner="Loading the demonstration portfolio…")
def get_portfolio_data(features: tuple[str, ...]) -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Portfolio file was not found: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    required_columns = set(features) | {TARGET}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Portfolio is missing required model columns: {sorted(missing_columns)}")
    if data.empty:
        raise ValueError("Portfolio does not contain any application rows.")
    if data[TARGET].isna().any() or not data[TARGET].isin([0, 1]).all():
        raise ValueError("Portfolio target must contain complete binary loan_status labels.")
    return data


try:
    artifact = get_model_artifact()
    feature_schema = get_model_input_schema(artifact)
except (FileNotFoundError, ValueError, OSError) as error:
    st.error(f"Could not load the fitted model: {error}")
    st.stop()

try:
    portfolio = get_portfolio_data(tuple(artifact["features"]))
except (FileNotFoundError, ValueError, OSError) as error:
    st.error(f"Could not load the portfolio: {error}")
    st.stop()

cutoffs = artifact["risk_band_cutoffs"]
target_rate = float(portfolio[TARGET].mean())
categorical_features = feature_schema["categorical"]

st.title("FinSight AI")
st.caption("Credit-risk modeling · educational dataset demo")
st.warning(
    "This is an educational model built from the supplied dataset, not a validated real-world "
    "probability of default. Do not use it to approve or reject credit, set lending terms, identify "
    "fraud, or make other consequential decisions. Do not enter real applicant information."
)

overview_tab, assessment_tab, performance_tab, explanation_tab, scenarios_tab = st.tabs(
    ["Portfolio overview", "Single-row demo", "Model details", "Model explanation", "Loss scenario"]
)

with overview_tab:
    st.subheader("Supplied portfolio")
    st.caption("Group rates describe observed dataset labels; they are descriptive, not causal.")
    default_count = int(portfolio[TARGET].sum())
    other_count = len(portfolio) - default_count
    metric_columns = st.columns(3)
    metric_columns[0].metric("Applications", f"{len(portfolio):,}")
    metric_columns[1].metric("Label-1 rate", f"{target_rate:.1%}")
    metric_columns[2].metric("Rows labelled 1", f"{default_count:,}")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(
            ["Label 0", "Label 1"],
            [other_count, default_count],
            color=["#4C78A8", "#E45756"],
        )
        ax.set_ylabel("Applications")
        ax.set_title("Observed target distribution")
        for bar, count in zip(bars, [other_count, default_count]):
            ax.text(bar.get_x() + bar.get_width() / 2, count, f"{count:,}", ha="center", va="bottom")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    if categorical_features:
        with chart_right:
            selected_segment = st.selectbox("Compare observed label-1 rates by", categorical_features)
            minimum_group_size = st.slider(
                "Minimum applications per category", 10, 1_000, 30, 10
            )
            segment_rates = summarize_default_rates(
                portfolio, selected_segment, target=TARGET, minimum_group_size=minimum_group_size
            ).sort_values("observed_default_rate")
            fig, ax = plt.subplots(figsize=(8, max(4, min(9, 0.35 * len(segment_rates)))))
            ax.barh(
                segment_rates["category"],
                segment_rates["observed_default_rate_percent"],
                color="#4C78A8",
            )
            ax.axvline(target_rate * 100, color="#E45756", linestyle="--", label="Portfolio rate")
            ax.set_xlabel("Observed label-1 rate (%)")
            ax.set_ylabel("")
            ax.set_title(f"Observed label rate by {selected_segment.replace('_', ' ')}")
            ax.legend(frameon=False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(
                segment_rates[
                    ["category", "applications", "observed_defaults", "observed_default_rate_percent"]
                ]
                .rename(columns={"observed_default_rate_percent": "observed rate (%)"})
                .round(2),
                width="stretch",
                hide_index=True,
            )

with assessment_tab:
    st.subheader("Single-row model demonstration")
    st.caption(
        "Enter values matching the model's training features. The supplied data is used for control "
        "ranges and categories; values submitted here are processed in memory only."
    )
    with st.form("model_input", clear_on_submit=False):
        input_columns = st.columns(2)
        inputs = {}
        for index, feature in enumerate(artifact["features"]):
            with input_columns[index % 2]:
                label = feature.replace("_", " ").title()
                if feature in feature_schema["categorical"]:
                    categories = sorted(portfolio[feature].dropna().astype(str).unique().tolist())
                    if not categories:
                        st.error(f"No categories are available for model input {feature}.")
                        st.stop()
                    inputs[feature] = st.selectbox(label, categories, key=f"input_{feature}")
                else:
                    values = pd.to_numeric(portfolio[feature], errors="raise").dropna()
                    if values.empty:
                        st.error(f"No numeric values are available for model input {feature}.")
                        st.stop()
                    minimum = float(values.min())
                    maximum = float(values.max())
                    default = min(max(float(values.median()), minimum), maximum)
                    if pd.api.types.is_integer_dtype(portfolio[feature].dtype):
                        inputs[feature] = st.number_input(
                            label,
                            min_value=int(np.floor(minimum)),
                            max_value=int(np.ceil(maximum)),
                            value=int(round(default)),
                            step=max(1, int(round((maximum - minimum) / 100))),
                            key=f"input_{feature}",
                        )
                    else:
                        step = max((maximum - minimum) / 100, 0.01)
                        inputs[feature] = st.number_input(
                            label,
                            min_value=minimum,
                            max_value=maximum,
                            value=default,
                            step=step,
                            format="%.4f",
                            key=f"input_{feature}",
                        )
        submitted = st.form_submit_button("Estimate dataset label probability", type="primary")

    if submitted:
        try:
            prediction = predict_application(inputs, artifact)
        except ValueError as error:
            st.error(f"Please correct the model inputs: {error}")
        else:
            probability = prediction["event_probability"]
            if probability < cutoffs["low_max"]:
                band = "Low demonstration band"
            elif probability < cutoffs["medium_max"]:
                band = "Medium demonstration band"
            else:
                band = "High demonstration band"
            result_columns = st.columns(3)
            result_columns[0].metric("Estimated probability of label 1", f"{probability:.1%}")
            result_columns[1].metric("Configurable display band", band)
            result_columns[2].metric(
                "Decision threshold comparison",
                "Above threshold" if prediction["predicted_label"] else "Below threshold",
            )
            st.info(
                f"The model estimates the probability of the dataset label "
                f"({prediction['event_definition']}). This output is not an approval, rejection, "
                "or recommendation."
            )

with performance_tab:
    st.subheader("Saved model configuration")
    st.caption(
        "The current model artifact contains its fitted classifier and thresholds, but not held-out "
        "evaluation metrics. No performance metrics are inferred from the full portfolio."
    )
    model = artifact["model"]
    calibrated_models = getattr(model, "calibrated_classifiers_", [])
    estimator = getattr(calibrated_models[0], "estimator", None) if calibrated_models else None
    classifier = estimator.named_steps.get("classifier") if hasattr(estimator, "named_steps") else None
    model_name = f"Calibrated {type(classifier).__name__}" if classifier is not None else type(model).__name__
    metric_columns = st.columns(4)
    metric_columns[0].metric("Estimator", model_name)
    metric_columns[1].metric("Saved threshold", f"{float(artifact['threshold']):.1%}")
    metric_columns[2].metric("Low-band upper limit", f"{float(cutoffs['low_max']):.1%}")
    metric_columns[3].metric("Medium-band upper limit", f"{float(cutoffs['medium_max']):.1%}")
    st.write("**Input features:**", ", ".join(artifact["features"]))
    st.markdown("Recorded notebook evaluation and experiment details: [model evaluation report](../reports/model_evaluation.md).")
    st.info("A saved threshold and model configuration do not establish suitability for real-world decisions.")

with explanation_tab:
    st.subheader("Global model behavior")
    st.caption(
        "Feature importance summarizes model reliance across calibration folds. It is not causal, "
        "does not explain an individual estimate, and is not a fairness assessment."
    )
    try:
        importance = get_global_feature_importance(artifact).head(15).sort_values(
            "mean_feature_importance"
        )
    except ValueError as error:
        st.error(f"Could not extract global feature importance from this model: {error}")
    else:
        fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(importance))))
        ax.barh(importance["feature"], importance["mean_feature_importance"], color="#4C78A8")
        ax.set_xlabel("Mean model feature importance")
        ax.set_title("Global model feature importance")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(
            importance.sort_values("mean_feature_importance", ascending=False),
            hide_index=True,
            width="stretch",
        )
    st.warning(
        "Correlated fields can share or distort feature importance. The chart does not explain "
        "individual predictions or demonstrate that predictors are appropriate for real lending."
    )

with scenarios_tab:
    st.subheader("Hypothetical expected-loss scenario")
    st.caption("Illustrative assumptions only; these values are not learned by the model.")
    st.latex(
        r"\mathrm{Expected\ Loss} = \mathrm{Event\ Probability} "
        r"\times \mathrm{LGD\ Assumption} \times \mathrm{EAD\ Assumption}"
    )
    left, right = st.columns(2)
    with left:
        event_probability = st.slider("Assumed event probability (%)", 0.0, 100.0, 8.0, 0.5) / 100
        lgd_assumption = st.slider("Assumed loss given default (%)", 0.0, 100.0, 45.0, 1.0) / 100
    with right:
        exposure = st.number_input(
            "Hypothetical exposure (arbitrary currency units)",
            min_value=0.0,
            max_value=1_000_000_000.0,
            value=100_000.0,
            step=10_000.0,
        )
    expected_loss = calculate_expected_loss(event_probability, lgd_assumption, exposure)
    st.metric("Illustrative expected loss", f"{expected_loss:,.2f}")
    st.info(
        "Probability, loss given default, and exposure are editable assumptions. The supplied target "
        "does not establish recovery rates or exposure at default."
    )

st.divider()
st.caption(
    "Educational dataset demo · inputs and predictions are not written to a database or file · "
    "do not enter real personal or financial information."
)
