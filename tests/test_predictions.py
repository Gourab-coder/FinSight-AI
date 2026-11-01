import unittest

import numpy as np

from src.predict import (
    get_model_input_schema,
    load_model_artifact,
    predict_application,
    prepare_model_input,
)


def valid_application():
    return {
        "person_age": 27,
        "person_income": 60_000,
        "person_home_ownership": "RENT",
        "person_emp_length": 5.0,
        "loan_intent": "PERSONAL",
        "loan_grade": "B",
        "loan_amnt": 10_000,
        "loan_int_rate": 11.14,
        "loan_percent_income": 0.17,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 5,
    }


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = load_model_artifact()

    def test_artifact_schema_matches_model_features(self):
        schema = get_model_input_schema(self.artifact)
        self.assertEqual(
            set(schema["numeric"] + schema["categorical"]),
            set(self.artifact["features"]),
        )
        self.assertEqual(
            list(prepare_model_input(valid_application(), self.artifact).columns),
            self.artifact["features"],
        )

    def test_prediction_probability_and_threshold_label_are_valid(self):
        prediction = predict_application(valid_application(), self.artifact)
        self.assertTrue(0 <= prediction["event_probability"] <= 1)
        self.assertEqual(prediction["decision_threshold"], self.artifact["threshold"])
        self.assertEqual(
            prediction["predicted_label"],
            int(prediction["event_probability"] >= self.artifact["threshold"]),
        )
        self.assertEqual(prediction["event_definition"], "loan_status = 1 (default label)")

    def test_missing_and_unexpected_inputs_are_rejected(self):
        application = valid_application()
        application.pop("person_income")
        with self.assertRaisesRegex(ValueError, "missing"):
            prepare_model_input(application, self.artifact)

        application = valid_application()
        application["unexpected_feature"] = 1
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            prepare_model_input(application, self.artifact)

    def test_non_finite_numeric_input_is_rejected(self):
        application = valid_application()
        application["person_income"] = np.inf
        with self.assertRaisesRegex(ValueError, "finite number"):
            prepare_model_input(application, self.artifact)

    def test_empty_categorical_input_is_rejected(self):
        application = valid_application()
        application["loan_intent"] = " "
        with self.assertRaisesRegex(ValueError, "non-empty category"):
            prepare_model_input(application, self.artifact)


if __name__ == "__main__":
    unittest.main()
