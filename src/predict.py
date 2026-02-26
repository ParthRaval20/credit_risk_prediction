import os
import sys
from typing import List

import joblib
import pandas as pd


MODEL_PATH = "credit_risk_model.pkl"
DATA_PATH = "data/credit_data.csv"
TARGET_COLUMN = "credit_risk"
# Match actual column names in credit_data.csv
REQUIRED_FEATURES: List[str] = [
    "age",
    "amount",
    "duration",
    "savings",
    "employment_duration",
]


def load_model(path: str):
    """Load the trained model with basic safety checks."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {os.path.abspath(path)}")

    try:
        model = joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load model from '{path}': {exc}") from exc

    if not hasattr(model, "predict_proba"):
        raise AttributeError("Loaded model does not implement 'predict_proba'.")

    return model


def load_data(path: str) -> pd.DataFrame:
    """Load the reference dataset used to build a template row."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {os.path.abspath(path)}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to read CSV data from '{path}': {exc}") from exc

    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' is missing from data. "
            f"Available columns: {list(df.columns)}"
        )

    return df


def build_new_customer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a single new-customer row based on an existing row
    and then override selected features.
    """
    features_df = df.drop(TARGET_COLUMN, axis=1)

    missing_features = [col for col in REQUIRED_FEATURES if col not in features_df.columns]
    if missing_features:
        raise KeyError(
            f"Required feature column(s) missing from data: {missing_features}. "
            f"Available columns: {list(features_df.columns)}"
        )

    if features_df.empty:
        raise ValueError("The feature data is empty; cannot build a template customer row.")

    # Take first row as a template
    new_customer = features_df.iloc[[0]].copy()

    # Modify values (simulate new person)
    new_customer["age"] = 30
    new_customer["amount"] = 3000
    new_customer["duration"] = 24
    new_customer["savings"] = "... < 100 DM"
    new_customer["employment_duration"] = "1 <= ... < 4 years"

    return new_customer


def predict_repayment_probability(model, customer_row: pd.DataFrame) -> float:
    """Predict repayment probability for a single customer row DataFrame."""
    if customer_row.shape[0] != 1:
        raise ValueError(
            f"Expected a single-row DataFrame, got {customer_row.shape[0]} rows instead."
        )

    try:
        # Model should return an array-like of shape (n_samples, 2)
        prob_safe = float(model.predict_proba(customer_row)[0][1])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Model prediction failed: {exc}") from exc

    return prob_safe


def main(threshold: float = 0.7) -> int:
    """Main entry point for running a prediction from the CLI."""
    try:
        model = load_model(MODEL_PATH)
        df = load_data(DATA_PATH)
        new_customer = build_new_customer(df)
        prob_safe = predict_repayment_probability(model, new_customer)
    except (FileNotFoundError, KeyError, ValueError, AttributeError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Repayment Probability: {prob_safe:.4f}")

    if prob_safe >= threshold:
        print("Loan Approved")
    else:
        print("Loan Rejected (High Risk)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())