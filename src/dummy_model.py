import numpy as np


class DummyCreditModel:
    """Simple dummy model that always returns a fixed safe probability."""

    def predict_proba(self, X):
        # X is expected to be an array-like / DataFrame with shape (n_samples, n_features)
        n_samples = len(X)
        # Fixed probability for the "safe" class
        p_safe = 0.8
        p_risky = 1.0 - p_safe
        # Return shape (n_samples, 2): [p_risky, p_safe]
        probs = np.tile([p_risky, p_safe], (n_samples, 1))
        return probs

