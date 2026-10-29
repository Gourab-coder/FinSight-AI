# Model evaluation

## Intended use and target

The selected model estimates the probability of the supplied dataset label,
`target_default_within_12m = 1`. It is an educational demonstration trained on
synthetic data, not a validated estimate of a real person's probability of
default. It must not be used to approve or reject credit, set lending terms,
identify fraud, or make other consequential decisions.

## Data and experiment design

- Dataset: 5,000 synthetic application records.
- Target prevalence: 27.68% positive (1,384 of 5,000).
- Split: stratified, reproducible train/validation/test partitions of 3,500,
  750, and 750 records (70% / 15% / 15%).
- Preprocessing and candidate tuning were fitted using training data only.
  Model and threshold selection used validation data. The held-out test set
  was evaluated after selection.
- Candidate estimators: Logistic Regression, Random Forest, and
  scikit-learn Histogram Gradient Boosting. XGBoost was not installed or
  declared as a project dependency, so it was not included.
- Selection: Random Forest was selected using validation average precision.
  Its saved hyperparameters are `max_depth=10`, `max_features="sqrt"`, and
  `min_samples_leaf=8`.
- Calibration: sigmoid calibration with 3-fold stratified cross-validation.
- Operating threshold: 0.23, selected using validation F1. This is a
  demonstration operating point, not a business policy.
- Feature engineering: `requested_loan_to_annual_income` is
  `requested_loan_amount_inr / (12 * monthly_net_income_inr)`. EMI and
  post-loan debt-to-income are computed from the documented application
  inputs.

## Held-out test results

| Metric | Result |
|---|---:|
| ROC-AUC | 0.7335 |
| Average precision | 0.5175 |
| Brier score | 0.1737 |
| Log loss | 0.5242 |
| Decision threshold | 0.2300 |
| Precision at threshold | 0.4377 |
| Recall at threshold | 0.6923 |
| F1 at threshold | 0.5363 |
| Applications flagged | 329 / 750 |

These are measurements on one synthetic held-out split. They are not evidence
of real-world calibration, production performance, or fitness for lending.
The threshold trades off false positives and false negatives; the selected
point is not inherently optimal outside this experiment.

## Interpretation and limitations

The dashboard reports global impurity-based Random Forest feature importance,
averaged across calibration folds and grouped back to original input fields.
The largest grouped importances in this fitted model are post-loan
debt-to-income, requested-loan-to-annual-income, and estimated EMI. These
values indicate model reliance, not causal effects, fairness, or explanations
for an individual prediction. Correlated fields can share or distort
impurity-based importance.

The data contain synthetic patterns and known data-quality limitations. In
particular, the observation window for `prior_default_flag` is undocumented,
and repayment timing is unclear; both are excluded from model inputs.
Identifiers, the target, and scenario-only LGD/EAD assumptions are also
excluded. The synthetic population, target-generation process, and single
random split do not establish representativeness, stability, fairness,
calibration drift, or legal suitability.

## Reproduction

Run Notebook 5 (`notebooks/05_supervised_modeling.ipynb`) from the project
environment to reproduce the experiment and regenerate
`models/credit_risk_model.joblib`. The saved artifact includes the final
pipeline, input contract, threshold, hyperparameters, and test metrics. Use
only trusted artifacts: Joblib deserialization can execute code.

For data-quality review, split rationale, candidate comparison, calibration
plots, and threshold analysis, see Notebook 5 and the earlier notebooks.
