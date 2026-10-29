"""Streamlit dashboard for the synthetic FinSight AI credit-risk project."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.predict import calculate_monthly_emi, load_model_artifact, predict_application
from src.risk_analytics import calculate_expected_loss, get_global_feature_importance, summarize_default_rates

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "finsight_ai_india_synthetic_5000.csv"
TARGET = "target_default_within_12m"
MODEL_PATH = PROJECT_ROOT / "models" / "credit_risk_model.joblib"

st.set_page_config(page_title="FinSight AI | Credit Risk Lab", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner="Loading the trusted fitted model…")
def get_model_artifact():
    return load_model_artifact(MODEL_PATH)


@st.cache_data(show_spinner="Loading the synthetic portfolio…")
def get_portfolio_data() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Synthetic portfolio file was not found: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    if TARGET not in data:
        raise ValueError(f"Synthetic portfolio is missing its expected target column: {TARGET}")
    if data[TARGET].isna().any() or not data[TARGET].isin([0, 1]).all():
        raise ValueError("Synthetic portfolio target must contain complete binary labels.")
    return data


try:
    artifact = get_model_artifact()
except FileNotFoundError as error:
    st.error(str(error))
    st.info("Run Notebook 5 and place the trusted fitted artifact in the project models/ directory.")
    st.stop()

try:
    portfolio = get_portfolio_data()
except (FileNotFoundError, ValueError, OSError) as error:
    st.error(f"Could not load the synthetic portfolio: {error}")
    st.stop()

st.title("FinSight AI")
st.caption("Credit-risk intelligence · synthetic-data educational demo")
st.warning(
    "This demonstration uses synthetic records. Model scores describe the chance of the dataset's "
    "12-month default label, not a validated real-world probability. Outputs are not loan approvals, "
    "rejections, fraud findings, or financial advice."
)
st.sidebar.header("Demonstration settings")
st.sidebar.caption("Risk bands below are configurable display thresholds, not lending policy.")
low_risk_max = st.sidebar.slider("Low-band upper limit", 0.05, 0.50, 0.20, 0.01)
medium_risk_max = st.sidebar.slider("Medium-band upper limit", 0.10, 0.95, 0.40, 0.01)

overview_tab, assessment_tab, performance_tab, explanation_tab, scenarios_tab = st.tabs(
    ["Portfolio overview", "Borrower assessment", "Model performance", "Model explanation", "Loss scenarios"]
)

with overview_tab:
    st.subheader("Synthetic portfolio")
    st.caption("Descriptive statistics over the supplied synthetic applications; subgroup rates are not causal.")
    default_count = int(portfolio[TARGET].sum())
    positive_rate = float(portfolio[TARGET].mean())
    no_default_count = len(portfolio) - default_count
    metric_columns = st.columns(4)
    metric_columns[0].metric("Applications", f"{len(portfolio):,}")
    metric_columns[1].metric("Observed label rate", f"{positive_rate:.1%}")
    metric_columns[2].metric("Default-labelled rows", f"{default_count:,}")
    metric_columns[3].metric("Other labelled rows", f"{no_default_count:,}")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(
            ["No default (0)", "Default (1)"],
            [no_default_count, default_count],
            color=["#4C78A8", "#E45756"],
        )
        ax.set_ylabel("Applications")
        ax.set_title("Observed 12-month target distribution")
        for bar, count in zip(bars, [no_default_count, default_count]):
            ax.text(bar.get_x() + bar.get_width() / 2, count, f"{count:,}", ha="center", va="bottom")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    category_fields = [
        "employment_type",
        "city_tier",
        "education_level",
        "loan_purpose",
        "state",
    ]
    with chart_right:
        selected_segment = st.selectbox("Compare observed rates by", category_fields)
        minimum_group_size = st.slider("Minimum applications per category", 10, 200, 30, 10)
        segment_rates = summarize_default_rates(
            portfolio, selected_segment, target=TARGET, minimum_group_size=minimum_group_size
        ).sort_values("observed_default_rate")
        fig, ax = plt.subplots(figsize=(8, max(4, min(9, 0.35 * len(segment_rates)))))
        ax.barh(
            segment_rates["category"],
            segment_rates["observed_default_rate_percent"],
            color="#4C78A8",
        )
        ax.axvline(positive_rate * 100, color="#E45756", linestyle="--", label="Portfolio rate")
        ax.set_xlabel("Observed default-labelled rows (%)")
        ax.set_ylabel("")
        ax.set_title(f"Observed label rate by {selected_segment.replace('_', ' ')}")
        ax.legend(frameon=False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.dataframe(
        segment_rates[["category", "applications", "observed_defaults", "observed_default_rate_percent"]]
        .rename(columns={"observed_default_rate_percent": "observed rate (%)"})
        .round(2),
        width="stretch",
        hide_index=True,
    )

with assessment_tab:
    st.subheader("Hypothetical borrower assessment")
    st.caption(
        "Enter application-time facts and proposed terms for a demonstration estimate. EMI and debt-to-income "
        "are calculated consistently from your inputs. Values are used in memory for this session only."
    )
    categories = {
        column: sorted(portfolio[column].dropna().astype(str).unique().tolist())
        for column in artifact["raw_feature_columns"]
        if column in {"state", "city_tier", "employment_type", "education_level", "residence_type", "loan_purpose"}
    }

    with st.form("borrower_assessment", clear_on_submit=False):
        st.markdown("#### Applicant and income")
        left, right = st.columns(2)
        with left:
            age_years = st.number_input("Applicant age (years)", min_value=18, max_value=100, value=35)
            employment_years = st.number_input(
                "Employment duration (years)", min_value=0.0, max_value=float(age_years), value=min(8.0, float(age_years)), step=0.5
            )
            state = st.selectbox("State / territory", categories["state"])
            city_tier = st.selectbox("City tier", categories["city_tier"])
            employment_type = st.selectbox("Employment type", categories["employment_type"])
            education_level = st.selectbox("Education", categories["education_level"])
            residence_type = st.selectbox("Residence type", categories["residence_type"])
        with right:
            monthly_income = st.number_input("Monthly net income (INR)", min_value=1, max_value=100_000_000, value=60_000, step=1_000)
            existing_debt = st.number_input("Existing monthly debt (INR)", min_value=0, max_value=100_000_000, value=12_000, step=1_000)
            savings = st.number_input("Savings balance (INR)", min_value=0, max_value=1_000_000_000, value=80_000, step=5_000)
            banking_months = st.number_input("Banking relationship (months)", min_value=0, max_value=1_200, value=80, step=1)
            dependents = st.number_input("Number of dependents", min_value=0, max_value=50, value=2, step=1)

        st.markdown("#### Requested loan and credit history")
        left, right = st.columns(2)
        with left:
            loan_amount = st.number_input("Requested loan amount (INR)", min_value=1_000, max_value=100_000_000, value=300_000, step=10_000)
            tenure_months = st.number_input("Loan tenure (months)", min_value=1, max_value=600, value=36, step=1)
            annual_interest = st.number_input("Annual interest rate (%)", min_value=0.0, max_value=100.0, value=16.0, step=0.25)
            loan_purpose = st.selectbox("Loan purpose", categories["loan_purpose"])
        with right:
            credit_history = st.number_input("Credit history (years)", min_value=0.0, max_value=float(age_years), value=min(7.0, float(age_years)), step=0.5)
            open_accounts = st.number_input("Open credit accounts", min_value=0, max_value=100, value=3, step=1)
            credit_inquiries = st.number_input("Credit inquiries in past 6 months", min_value=0, max_value=100, value=1, step=1)
            late_payments = st.number_input("Late payments in past 12 months", min_value=0, max_value=100, value=0, step=1)
            utilization = st.number_input("Credit utilization (ratio; 1.0 = 100%)", min_value=0.0, max_value=10.0, value=0.35, step=0.01)

        submitted = st.form_submit_button("Estimate dataset-event probability", type="primary")

    if submitted:
        if medium_risk_max <= low_risk_max:
            st.error("Set the medium-band upper limit higher than the low-band upper limit in the sidebar.")
        else:
            application_inputs = {
                "age_years": age_years,
                "employment_years": employment_years,
                "monthly_net_income_inr": monthly_income,
                "existing_monthly_debt_inr": existing_debt,
                "requested_loan_amount_inr": loan_amount,
                "loan_tenure_months": tenure_months,
                "annual_interest_rate_pct": annual_interest,
                "credit_history_years": credit_history,
                "open_credit_accounts": open_accounts,
                "credit_inquiries_last_6m": credit_inquiries,
                "late_payments_last_12m": late_payments,
                "credit_utilization_ratio": utilization,
                "savings_balance_inr": savings,
                "banking_relationship_months": banking_months,
                "dependents_count": dependents,
                "state": state,
                "city_tier": city_tier,
                "employment_type": employment_type,
                "education_level": education_level,
                "residence_type": residence_type,
                "loan_purpose": loan_purpose,
            }
            try:
                prediction = predict_application(application_inputs, artifact)
            except ValueError as error:
                st.error(f"Please correct the application inputs: {error}")
            else:
                probability = prediction["event_probability"]
                if probability < low_risk_max:
                    band = "Low demonstration band"
                elif probability < medium_risk_max:
                    band = "Medium demonstration band"
                else:
                    band = "High demonstration band"
                result_columns = st.columns(3)
                result_columns[0].metric("Estimated event probability", f"{probability:.1%}")
                result_columns[1].metric("Configurable display band", band)
                result_columns[2].metric(
                    "Threshold comparison",
                    "Above demonstration threshold" if prediction["predicted_label"] else "Below demonstration threshold",
                )
                monthly_emi = calculate_monthly_emi(loan_amount, annual_interest, tenure_months)
                debt_to_income = (existing_debt + monthly_emi) / monthly_income
                st.caption(
                    f"Derived EMI: INR {monthly_emi:,.0f} per month · "
                    f"derived post-loan debt-to-income: {debt_to_income:.1%} · "
                    f"saved model threshold: {prediction['decision_threshold']:.1%}"
                )
                st.info(
                    f"This model estimates the probability of the dataset event "
                    f"({prediction['event_definition']}). It is not a probability that any particular "
                    "loan will default and is not an approval, rejection, or fraud recommendation."
                )

with performance_tab:
    st.subheader("Held-out evaluation")
    st.caption(
        "These metrics were saved by Notebook 5 from one stratified test split. Model and threshold choices "
        "were made on training/validation data before the test set was evaluated."
    )
    metrics = artifact["test_metrics"]
    metric_columns = st.columns(4)
    for column, label, key, format_string in [
        (metric_columns[0], "ROC-AUC", "roc_auc", "{:.3f}"),
        (metric_columns[1], "Average precision", "average_precision", "{:.3f}"),
        (metric_columns[2], "Brier score", "brier_score", "{:.3f}"),
        (metric_columns[3], "Log loss", "log_loss", "{:.3f}"),
    ]:
        column.metric(label, format_string.format(float(metrics[key])))
    operating_columns = st.columns(4)
    operating_columns[0].metric("Saved threshold", f"{float(artifact['decision_threshold']):.1%}")
    operating_columns[1].metric("Test precision", f"{float(metrics['precision_at_threshold']):.1%}")
    operating_columns[2].metric("Test recall", f"{float(metrics['recall_at_threshold']):.1%}")
    operating_columns[3].metric("Test F1", f"{float(metrics['f1_at_threshold']):.3f}")
    st.write(
        f"At the validation-selected operating point, the model flagged "
        f"{int(metrics['alerts_at_threshold']):,} of {int(metrics['test_applications']):,} held-out applications."
    )
    st.write("**Selected model:**", artifact["model_name"])
    st.write("**Calibrator:**", artifact["calibration_method"])
    st.write("**Validation tuning metric:**", artifact["validation_model_selection_metric"])
    st.write("**Validation threshold-selection metric:**", artifact["threshold_selection_metric"])
    st.write("**Selected hyperparameters:**")
    st.json(artifact["selected_hyperparameters"])
    st.caption("See Notebook 5 for candidate-model comparisons, threshold curves, confusion matrix, and calibration plots.")
    st.info("This is a single synthetic-data holdout evaluation; it does not establish production generalization.")

with explanation_tab:
    st.subheader("Global model behavior")
    st.caption(
        "The selected Random Forest's built-in impurity importances are averaged across the fitted "
        "calibration folds. These describe model reliance, not causal effects, individual reasons, or fairness."
    )
    try:
        importance = get_global_feature_importance(artifact).head(15).sort_values("mean_impurity_importance")
    except ValueError as error:
        st.error(f"Could not extract global feature importance from this model: {error}")
    else:
        fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(importance))))
        ax.barh(importance["feature"], importance["mean_impurity_importance"], color="#4C78A8")
        ax.set_xlabel("Mean impurity-based importance (relative)")
        ax.set_title("Global model feature importance")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(
            importance.sort_values("mean_impurity_importance", ascending=False),
            hide_index=True,
            width="stretch",
        )
    st.warning(
        "Correlated fields can share or distort tree importance. This chart is not SHAP, does not explain an "
        "individual estimate, and does not demonstrate that predictors are appropriate for real lending."
    )

with scenarios_tab:
    st.subheader("Hypothetical expected-loss scenario")
    st.caption("Enter explicit assumptions. None of these scenario values is learned from the available dataset.")
    st.latex(r"\mathrm{Expected\ Loss} = \mathrm{Event\ Probability} \times \mathrm{LGD\ Assumption} \times \mathrm{EAD\ Assumption}")
    left, right = st.columns(2)
    with left:
        event_probability = st.slider("Assumed event probability (%)", 0.0, 100.0, 8.0, 0.5) / 100
        lgd_assumption = st.slider("Assumed loss given default (%)", 0.0, 100.0, 45.0, 1.0) / 100
    with right:
        exposure = st.number_input("Hypothetical exposure at default (INR)", min_value=0.0, max_value=1_000_000_000.0, value=100_000.0, step=10_000.0)
    expected_loss = calculate_expected_loss(event_probability, lgd_assumption, exposure)
    st.metric("Illustrative expected loss", f"INR {expected_loss:,.2f}")
    st.info(
        "Illustrative scenario only: probability, LGD, and EAD are editable assumptions. The dataset target "
        "is not a universally valid probability of default, and the available data do not establish actual "
        "recovery rates or exposure at default."
    )

st.divider()
st.caption(
    "Synthetic educational project · no borrower inputs or predictions are written to a database or file · "
    "do not enter real personal or financial information."
)
