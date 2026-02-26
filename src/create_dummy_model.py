import joblib

from dummy_model import DummyCreditModel


def main() -> None:
    model = DummyCreditModel()
    joblib.dump(model, "credit_risk_model.pkl")
    print("Dummy credit risk model saved to 'credit_risk_model.pkl'.")


if __name__ == "__main__":
    main()

