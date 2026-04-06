from sklearn.ensemble import RandomForestClassifier

from classificators.feature_based_ensemble_classifier import FeatureBasedEnsembleClassifier


class EnhancedRandomForestClassifier(FeatureBasedEnsembleClassifier):
    """Random-forest variant of the shared feature-based one-vs-rest pipeline."""

    def __init__(
        self,
        classes,
        sensor_names,
        n_estimators=200,
        max_train_samples=50000,
        random_state=42,
        batch_size=50000,
    ):
        super().__init__(
            classes=classes,
            sensor_names=sensor_names,
            max_train_samples=max_train_samples,
            random_state=random_state,
            batch_size=batch_size,
            model_name="enhanced RF",
        )
        self.n_estimators = n_estimators

    def _build_model(self):
        """Create one binary random-forest classifier."""
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            n_jobs=-1,
            random_state=self.random_state,
            class_weight="balanced_subsample",
        )
