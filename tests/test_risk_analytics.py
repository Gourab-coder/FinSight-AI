import unittest

import pandas as pd

from src.predict import load_model_artifact
from src.risk_analytics import (
    calculate_expected_loss,
    get_global_feature_importance,
    summarize_default_rates,
)


class RiskAnalyticsTests(unittest.TestCase):
    def test_expected_loss_uses_explicit_probability_lgd_and_ead(self):
        self.assertAlmostEqual(calculate_expected_loss(0.08, 0.45, 100_000), 3_600)

    def test_expected_loss_rejects_out_of_range_fractions(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            calculate_expected_loss(1.1, 0.45, 100_000)

    def test_expected_loss_rejects_negative_exposure(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            calculate_expected_loss(0.08, 0.45, -1)

    def test_group_rates_preserve_missing_category_and_counts(self):
        data = pd.DataFrame(
            {
                "segment": ["A", "A", "B", None],
                "loan_status": [1, 0, 1, 0],
            }
        )
        result = summarize_default_rates(data, "segment")
        missing_group = result.loc[result["category"] == "Missing / not recorded"].iloc[0]
        self.assertEqual(missing_group["applications"], 1)
        self.assertEqual(missing_group["observed_defaults"], 0)
        self.assertEqual(missing_group["observed_default_rate"], 0)

    def test_group_size_filter_is_explicit(self):
        data = pd.DataFrame(
            {
                "segment": ["A", "A", "B"],
                "loan_status": [1, 0, 1],
            }
        )
        result = summarize_default_rates(data, "segment", minimum_group_size=2)
        self.assertEqual(result["category"].tolist(), ["A"])

    def test_feature_importance_is_grouped_and_ordered(self):
        artifact = load_model_artifact()
        result = get_global_feature_importance(artifact)
        self.assertIn("feature", result)
        self.assertIn("mean_feature_importance", result)
        self.assertTrue(result["mean_feature_importance"].is_monotonic_decreasing)
        self.assertEqual(set(result["feature"]), set(artifact["features"]))


if __name__ == "__main__":
    unittest.main()
