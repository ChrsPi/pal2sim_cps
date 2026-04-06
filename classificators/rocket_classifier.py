import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from classificators.rocket_transform import RocketTransform
from classificators.window_features import ensure_channel_first_windows, sample_training_windows


class RocketWindowClassifier:
    """Torch ROCKET classifier with GPU-capable transform and linear multi-label head."""

    def __init__(
        self,
        classes,
        sensor_names,
        num_kernels=256,
        max_train_samples=50000,
        transform_batch_size=4096,
        train_batch_size=1024,
        num_epochs=20,
        learning_rate=1e-3,
        weight_decay=1e-4,
        pos_weight_power=0.25,
        device=None,
        random_state=42,
    ):
        self.classes = classes
        self.sensor_names = sensor_names
        self.num_kernels = num_kernels
        self.max_train_samples = max_train_samples
        self.transform_batch_size = transform_batch_size
        self.train_batch_size = train_batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.pos_weight_power = pos_weight_power
        self.random_state = random_state
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.transformer = RocketTransform(
            num_kernels=num_kernels,
            random_state=random_state,
        )
        self.model = None
        self.feature_mean = None
        self.feature_std = None
        self.thresholds = np.zeros(len(classes), dtype=np.float32)

    @staticmethod
    def _safe_mcc(y_true, y_pred):
        """Compute MCC while treating degenerate constant-label cases as zero."""
        if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
            return 0.0
        return matthews_corrcoef(y_true, y_pred)

    def _transform_to_cpu_tensor(self, X):
        """Transform windows in batches and return a CPU feature tensor."""
        X = ensure_channel_first_windows(X, len(self.sensor_names))
        feature_batches = []

        for start in range(0, len(X), self.transform_batch_size):
            end = min(start + self.transform_batch_size, len(X))
            features = self.transformer.transform(X[start:end], device=self.device)
            feature_batches.append(features.cpu())

        return torch.cat(feature_batches, dim=0)

    def _normalize_cpu_features(self, features):
        """Normalize CPU feature tensors using stored training statistics."""
        return (features - self.feature_mean.cpu()) / self.feature_std.cpu()

    def _score_matrix(self, X):
        """Predict per-class logits for arbitrary windows."""
        self.model.eval()
        X = ensure_channel_first_windows(X, len(self.sensor_names))
        score_batches = []

        with torch.no_grad():
            for start in range(0, len(X), self.transform_batch_size):
                end = min(start + self.transform_batch_size, len(X))
                features = self.transformer.transform(X[start:end], device=self.device)
                features = (features - self.feature_mean) / self.feature_std
                logits = self.model(features)
                score_batches.append(logits.cpu().numpy())

        return np.vstack(score_batches)

    def _optimize_thresholds(self, y_true, scores):
        """Choose a validation logit threshold per class that maximizes MCC."""
        for class_idx, class_name in enumerate(self.classes):
            class_scores = scores[:, class_idx]
            candidate_thresholds = np.unique(
                np.concatenate(
                    [
                        np.array([0.0], dtype=np.float32),
                        np.quantile(class_scores, np.linspace(0.05, 0.95, 19)).astype(np.float32),
                    ]
                )
            )

            best_threshold = 0.0
            best_mcc = -np.inf

            for threshold in candidate_thresholds:
                predictions = (class_scores >= threshold).astype(int)
                score = self._safe_mcc(y_true[:, class_idx], predictions)

                if score > best_mcc or (
                    np.isclose(score, best_mcc) and
                    abs(threshold) < abs(best_threshold)
                ):
                    best_mcc = score
                    best_threshold = float(threshold)

            self.thresholds[class_idx] = best_threshold
            print(
                f"Validation threshold for {class_name}: "
                f"{best_threshold:.4f} (MCC={best_mcc:.4f})"
            )

    def train(self, train_data, val_data):
        """Fit the random-kernel transform, train a torch linear head, and tune thresholds."""
        torch.manual_seed(self.random_state)

        X_train, y_train = sample_training_windows(
            train_data[0],
            train_data[1],
            max_samples=self.max_train_samples,
            random_state=self.random_state,
        )
        X_train = ensure_channel_first_windows(X_train, len(self.sensor_names))

        print(
            f"Fitting Torch ROCKET transform with {len(X_train)} samples, "
            f"{self.num_kernels} kernels on {self.device}..."
        )
        self.transformer.fit(X_train)
        X_train_features = self._transform_to_cpu_tensor(X_train)

        feature_mean = X_train_features.mean(dim=0, keepdim=True)
        feature_std = X_train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
        self.feature_mean = feature_mean.to(self.device)
        self.feature_std = feature_std.to(self.device)
        X_train_features = self._normalize_cpu_features(X_train_features)

        y_train_tensor = torch.as_tensor(y_train, dtype=torch.float32)
        dataset = TensorDataset(X_train_features, y_train_tensor)
        loader = DataLoader(dataset, batch_size=self.train_batch_size, shuffle=True)

        self.model = nn.Linear(X_train_features.shape[1], len(self.classes)).to(self.device)

        positive_counts = y_train_tensor.sum(dim=0)
        negative_counts = len(y_train_tensor) - positive_counts
        pos_weight = torch.pow(
            negative_counts / positive_counts.clamp_min(1.0),
            self.pos_weight_power,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_features, batch_targets in loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_features)
                loss = criterion(logits, batch_targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(batch_features)

            if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == self.num_epochs - 1:
                avg_loss = epoch_loss / len(dataset)
                print(f"Epoch {epoch + 1}/{self.num_epochs} - loss={avg_loss:.4f}")

        val_scores = self._score_matrix(val_data[0])
        self._optimize_thresholds(val_data[1], val_scores)
        print("Training done.")

    def predict(self, test_X):
        """Predict multi-hot labels from ROCKET logits."""
        scores = self._score_matrix(test_X)
        return (scores >= self.thresholds).astype(int)
