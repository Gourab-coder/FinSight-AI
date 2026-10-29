import unittest

import numpy as np

from src.predict import (
    calculate_monthly_emi,
    load_model_artifact,
    predict_application,
    prepare_model_input,
)


def valid_application():
    return {
        "age_years": 35,
        "employment_years": 8.0,
        "monthly_net_income_inr": 60_000,
        "existing_monthly_debt_inr": 12_000,
        "requested_loan_amount_inr": 300_000,
        "loan_tenure_months": 36,
        "annual_interest_rate_pct": 16.0,
        "credit_history_years": 7.0,
        "open_credit_accounts": 3,
        "credit_inquiries_last_6m": 1,
        "late_payments_last_12m": 0,
        "credit_utilization_ratio": 0.35,
        "savings_balance_inr": 80_000,
        "banking_relationship_months": 80,
        "dependents_count": 2,
        "state": "Delhi NCR",
        "city_tier": "Tier 1",
        "employment_type": "Salaried",
        "education_level": "Graduate",
        "residence_type": "Rented",
        "loan_purpose": "Vehicle",
    }


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = load_model_artifact()

    def test_zero_rate_emi_is_principal_divided_by_tenure(self):
        self.assertAlmostEqual(calculate_monthly_emi(120_000, 0, 12), 10_000)

    def test_positive_rate_emi_amortizes_the_requested_principal(self):
        payment = calculate_monthly_emi(300_000, 16, 36)
        self.assertGreater(payment, 300_000 / 36)
        self.assertTrue(np.isfinite(payment))

    def test_input_builder_derives_model_fields_and_respects_artifact_order(self):
        row = prepare_model_input(valid_application(), self.artifact)
        self.assertEqual(list(row.columns), self.artifact["model_input_columns"])
        expected_emi = calculate_monthly_emi(300_000, 16, 36)
        self.assertAlmostEqual(row.loc[0, "estimated_emi_inr"], expected_emi)
        self.assertAlmostEqual(
            row.loc[0, "debt_to_income_after_loan"], (12_000 + expected_emi) / 60_000
        )
        self.assertAlmostEqual(row.loc[0, "requested_loan_to_annual_income"], 300_000 / (12 * 60_000))

    def test_prediction_is_finite_and_uses_the_saved_threshold(self):
        prediction = predict_application(valid_application(), self.artifact)
        self.assertTrue(0 <= prediction["event_probability"] <= 1)
        self.assertEqual(prediction["decision_threshold"], self.artifact["decision_threshold"])
        self.assertEqual(
            prediction["predicted_label"],
            int(prediction["event_probability"] >= self.artifact["decision_threshold"]),
        )

    def test_invalid_income_is_rejected(self):
        application = valid_application()
        application["monthly_net_income_inr"] = 0
        with self.assertRaisesRegex(ValueError, "income"):
            prepare_model_input(application, self.artifact)

    def test_impossible_employment_duration_is_rejected(self):
        application = valid_application()
        application["employment_years"] = 36
        with self.assertRaisesRegex(ValueError, "Employment duration"):
            prepare_model_input(application, self.artifact)

    def test_unknown_or_misspelled_inputs_are_not_silently_ignored(self):
        application = valid_application()
        application["montly_net_income_inr"] = application.pop("monthly_net_income_inr")
        with self.assertRaisesRegex(ValueError, "missing"):
            prepare_model_input(application, self.artifact)


if __name__ == "__main__":
    unittest.main()
