import numpy as np
from sklearn.metrics import matthews_corrcoef


def safe_mcc(y_true, y_pred):
    """Compute MCC while treating degenerate constant-label cases as zero."""
    if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return 0.0
    return matthews_corrcoef(y_true, y_pred)


def sample_evenly(X, y=None, max_samples=None):
    """Sample windows evenly across a sequence to reduce overlap bias and runtime."""
    if max_samples is None or len(X) <= max_samples:
        return (X, y) if y is not None else X

    idx = np.linspace(0, len(X) - 1, num=max_samples, dtype=int)
    if y is None:
        return X[idx]
    return X[idx], y[idx]


def optimize_probability_thresholds(y_true, y_proba):
    """Choose one probability threshold per class to maximize validation MCC."""
    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)

    for class_idx in range(y_true.shape[1]):
        class_proba = y_proba[:, class_idx]
        candidate_thresholds = np.unique(
            np.concatenate(
                [
                    np.array([0.5], dtype=np.float32),
                    np.quantile(class_proba, np.linspace(0.05, 0.95, 19)).astype(np.float32),
                ]
            )
        )

        best_threshold = 0.5
        best_mcc = -np.inf

        for threshold in candidate_thresholds:
            predictions = (class_proba >= threshold).astype(int)
            score = safe_mcc(y_true[:, class_idx], predictions)

            if score > best_mcc or (
                np.isclose(score, best_mcc) and
                abs(threshold - 0.5) < abs(best_threshold - 0.5)
            ):
                best_mcc = score
                best_threshold = float(threshold)

        thresholds[class_idx] = best_threshold

    return thresholds
