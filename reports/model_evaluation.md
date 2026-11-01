# Model evaluation

## Intended use and target

The model estimates the probability of the supplied dataset label,
`loan_status = 1`. It is an educational experiment, not a validated estimate
of a real person's probability of default. Do not use it to approve or reject
credit, set lending terms, identify fraud, or make other consequential
decisions.

## Data and experiment design

- The notebook starts with 32,581 records, removes duplicate and invalid
  records, and retains 32,409 rows for modeling.
- Features are the 11 columns saved in the model artifact. `loan_status` is
  excluded from model inputs.
- The notebook uses stratified train, validation, and test partitions of
  approximately 60% / 20% / 20%, with `random_state=42`.
- Preprocessing and hyperparameter search are fitted on training data.
  Validation data is used for calibration checks and decision-threshold
  selection; the held-out test data is evaluated after selection.
- The notebook compares Logistic Regression with tuned XGBoost and selects
  calibrated XGBoost. The saved classification threshold is approximately
  0.633, selected to maximize validation F1.

## Held-out test results

The following values are from the executed notebook's final calibrated
XGBoost test evaluation, at the validation-selected threshold:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.9514 |
| Average precision (PR-AUC) | 0.9090 |
| Accuracy | 0.9372 |
| Precision | 0.9558 |
| Recall | 0.7475 |
| F1 | 0.8389 |
| Specificity | 0.9903 |
| Balanced accuracy | 0.8689 |
| Matthews correlation coefficient | 0.8100 |
| Brier score | 0.0528 |
| Decision threshold | 0.6333 |

These are measurements on one held-out split of this dataset. They are not
evidence of real-world calibration, production performance, representativeness,
or fitness for lending. The operating threshold trades off false positives and
false negatives and is not a business policy.

## Interpretation and limitations

The dashboard aggregates the fitted XGBoost feature importances across
calibration folds and groups encoded category values back to their original
input fields. These values describe model reliance, not causal effects,
fairness, or reasons for an individual prediction. Correlated fields can share
or distort feature importance.

The supplied dataset and its target-generation process may not represent
future applicants or any deployment population. The reported split does not
establish stability, fairness, calibration drift, legal suitability, or
generalization to real lending decisions. The dashboard's risk bands and
expected-loss inputs are illustrative settings, not validated lending policy
or learned recovery estimates.

## Reproduction

Run `notebooks/01_NoteBook_Credit_Risk_Project.ipynb` in the project
environment to reproduce the experiment and regenerate
`models/credit_risk_model.joblib`. The artifact contains the fitted classifier,
threshold, risk-band cutoffs, and ordered input feature names; it does not
contain held-out metric values. Use only trusted artifacts: Joblib
deserialization can execute code.
