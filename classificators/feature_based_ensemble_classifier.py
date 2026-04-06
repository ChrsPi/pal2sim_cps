import numpy as np
from sklearn.metrics import matthews_corrcoef

from classificators.window_features import extract_window_features, sample_training_windows


class FeatureBasedEnsembleClassifier:
    """Base class for one-vs-rest tree models trained on extracted window features."""

    def __init__(
        self,
        classes,
        sensor_names,
        max_train_samples=50000,
        random_state=42,
        batch_size=50000,
        model_name="feature ensemble",
    ):
        self.classes = classes
        self.sensor_names = sensor_names
        self.max_train_samples = max_train_samples
        self.random_state = random_state
        self.batch_size = batch_size
        self.model_name = model_name
        self.models = []
        self.thresholds = np.full(len(classes), 0.5, dtype=np.float32)

    def _build_model(self):
        """Create a fresh binary estimator for one target class."""
        raise NotImplementedError

    @staticmethod
    def _positive_class_probability(model, X_features):
        """Return the probability assigned to class ``1`` for a fitted binary model."""
        probabilities = model.predict_proba(X_features)
        if probabilities.shape[1] == 1:
            return np.full(len(X_features), float(model.classes_[0] == 1), dtype=np.float32)

        positive_index = np.where(model.classes_ == 1)[0]
        if len(positive_index) == 0:
            return np.zeros(len(X_features), dtype=np.float32)

        return probabilities[:, positive_index[0]].astype(np.float32, copy=False)

    def _predict_probability_matrix(self, X):
        """Predict class probabilities in batches for all one-vs-rest models."""
        probabilities = np.zeros((len(X), len(self.classes)), dtype=np.float32)

        for start in range(0, len(X), self.batch_size):
            end = min(start + self.batch_size, len(X))
            X_features = extract_window_features(X[start:end], self.sensor_names)

            for class_idx, model in enumerate(self.models):
                probabilities[start:end, class_idx] = self._positive_class_probability(model, X_features)

        return probabilities

    @staticmethod
    def _safe_mcc(y_true, y_pred):
        """Compute MCC while treating degenerate constant-label cases as zero."""
        if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
            return 0.0
        return matthews_corrcoef(y_true, y_pred)

    def _optimize_thresholds(self, y_true, y_proba):
        """Choose a per-class decision threshold that maximizes validation MCC."""
        candidate_thresholds = np.arange(0.10, 0.91, 0.05)

        for class_idx, class_name in enumerate(self.classes):
            best_threshold = 0.5
            best_mcc = -np.inf

            for threshold in candidate_thresholds:
                predictions = (y_proba[:, class_idx] >= threshold).astype(int)
                score = self._safe_mcc(y_true[:, class_idx], predictions)

                if score > best_mcc or (
                    np.isclose(score, best_mcc) and
                    abs(threshold - 0.5) < abs(best_threshold - 0.5)
                ):
                    best_mcc = score
                    best_threshold = threshold

            self.thresholds[class_idx] = best_threshold
            print(
                f"Validation threshold for {class_name}: "
                f"{best_threshold:.2f} (MCC={best_mcc:.4f})"
            )

    def train(self, train_data, val_data):
        """Fit one binary model per class and tune thresholds on validation data."""
        X_train, y_train = sample_training_windows(
            train_data[0],
            train_data[1],
            max_samples=self.max_train_samples,
            random_state=self.random_state,
        )
        X_features = extract_window_features(X_train, self.sensor_names)

        self.models = []
        print(
            f"Training {self.model_name} with {X_features.shape[0]} samples "
            f"and {X_features.shape[1]} features..."
        )

        for class_idx, class_name in enumerate(self.classes):
            model = self._build_model()
            model.fit(X_features, y_train[:, class_idx])
            self.models.append(model)
            print(f"Trained binary model for class: {class_name}")

        val_probabilities = self._predict_probability_matrix(val_data[0])
        self._optimize_thresholds(val_data[1], val_probabilities)
        print("Training done.")

    def predict(self, test_X):
        """Convert predicted probabilities into multi-hot class predictions."""
        probabilities = self._predict_probability_matrix(test_X)
        return (probabilities >= self.thresholds).astype(int)
