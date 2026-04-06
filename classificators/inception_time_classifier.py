import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from classificators.torch_multilabel_utils import optimize_probability_thresholds, sample_evenly, safe_mcc
from classificators.window_features import ensure_channel_first_windows, sample_training_windows


class InceptionModule(nn.Module):
    """One InceptionTime module with three convolution branches and one pooling branch."""

    def __init__(self, in_channels, bottleneck_channels=32, out_channels=32, kernel_sizes=(9, 19, 39)):
        super().__init__()
        self.use_bottleneck = in_channels > 1
        reduced_channels = bottleneck_channels if self.use_bottleneck else in_channels

        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            if self.use_bottleneck else nn.Identity()
        )
        self.conv_branches = nn.ModuleList(
            [
                nn.Conv1d(
                    reduced_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
        )
        self.batch_norm = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))
        self.activation = nn.ReLU()

    def forward(self, x):
        """Apply parallel temporal convolutions and concatenate their outputs."""
        bottleneck = self.bottleneck(x)
        conv_outputs = [branch(bottleneck) for branch in self.conv_branches]
        pool_output = self.pool_branch(x)
        merged = torch.cat(conv_outputs + [pool_output], dim=1)
        return self.activation(self.batch_norm(merged))


class InceptionResidualBlock(nn.Module):
    """Three stacked Inception modules plus a residual shortcut."""

    def __init__(self, in_channels, bottleneck_channels=32, out_channels=32):
        super().__init__()
        hidden_channels = out_channels * 4
        self.inception_1 = InceptionModule(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
        )
        self.inception_2 = InceptionModule(
            in_channels=hidden_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
        )
        self.inception_3 = InceptionModule(
            in_channels=hidden_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
        )
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        """Apply three modules then add a channel-matched shortcut."""
        residual = self.shortcut(x)
        x = self.inception_1(x)
        x = self.inception_2(x)
        x = self.inception_3(x)
        return self.activation(x + residual)


class InceptionTimeNetwork(nn.Module):
    """Compact InceptionTime network for multi-label window classification."""

    def __init__(self, in_channels, n_classes, bottleneck_channels=32, out_channels=32, n_blocks=2):
        super().__init__()
        blocks = []
        current_channels = in_channels

        for _ in range(n_blocks):
            block = InceptionResidualBlock(
                in_channels=current_channels,
                bottleneck_channels=bottleneck_channels,
                out_channels=out_channels,
            )
            blocks.append(block)
            current_channels = out_channels * 4

        self.backbone = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(current_channels, n_classes)

    def forward(self, x):
        """Map window tensors of shape ``(N, C, T)`` to class logits."""
        x = self.backbone(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


class InceptionTimeWindowClassifier:
    """Torch InceptionTime classifier with MCC-based threshold tuning."""

    def __init__(
        self,
        classes,
        sensor_names,
        max_train_samples=120000,
        max_val_samples=50000,
        train_batch_size=1024,
        inference_batch_size=4096,
        num_epochs=20,
        learning_rate=1e-3,
        weight_decay=1e-4,
        pos_weight_power=0.25,
        bottleneck_channels=32,
        out_channels=32,
        n_blocks=2,
        device=None,
        random_state=42,
    ):
        self.classes = classes
        self.sensor_names = sensor_names
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.train_batch_size = train_batch_size
        self.inference_batch_size = inference_batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.pos_weight_power = pos_weight_power
        self.bottleneck_channels = bottleneck_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.random_state = random_state
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = None
        self.thresholds = np.full(len(classes), 0.5, dtype=np.float32)

    def _make_loader(self, X, y=None, shuffle=False):
        """Create a torch dataloader from window arrays."""
        X = np.array(
            ensure_channel_first_windows(X, len(self.sensor_names)),
            dtype=np.float32,
            copy=True,
        )
        X_tensor = torch.from_numpy(X)

        if y is None:
            dataset = TensorDataset(X_tensor)
        else:
            y_tensor = torch.as_tensor(y, dtype=torch.float32)
            dataset = TensorDataset(X_tensor, y_tensor)

        return DataLoader(dataset, batch_size=self.train_batch_size if y is not None else self.inference_batch_size, shuffle=shuffle)

    def _predict_probabilities(self, X):
        """Predict class probabilities for arbitrary windows."""
        self.model.eval()
        loader = self._make_loader(X, y=None, shuffle=False)
        probabilities = []

        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                probabilities.append(torch.sigmoid(logits).cpu().numpy())

        return np.vstack(probabilities)

    def _print_threshold_summary(self, y_true, y_proba):
        """Tune and log per-class thresholds on validation probabilities."""
        self.thresholds = optimize_probability_thresholds(y_true, y_proba)

        for class_idx, class_name in enumerate(self.classes):
            predictions = (y_proba[:, class_idx] >= self.thresholds[class_idx]).astype(int)
            score = safe_mcc(y_true[:, class_idx], predictions)
            print(
                f"Validation threshold for {class_name}: "
                f"{self.thresholds[class_idx]:.4f} (MCC={score:.4f})"
            )

    def train(self, train_data, val_data):
        """Fit InceptionTime on sampled windows and tune thresholds on validation data."""
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        X_train, y_train = sample_training_windows(
            train_data[0],
            train_data[1],
            max_samples=self.max_train_samples,
            random_state=self.random_state,
        )
        X_val, y_val = sample_evenly(
            val_data[0],
            val_data[1],
            max_samples=self.max_val_samples,
        )

        loader = self._make_loader(X_train, y_train, shuffle=True)
        self.model = InceptionTimeNetwork(
            in_channels=len(self.sensor_names),
            n_classes=len(self.classes),
            bottleneck_channels=self.bottleneck_channels,
            out_channels=self.out_channels,
            n_blocks=self.n_blocks,
        ).to(self.device)

        positive_counts = torch.as_tensor(y_train, dtype=torch.float32).sum(dim=0)
        negative_counts = len(y_train) - positive_counts
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
        scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")

        print(
            f"Training InceptionTime with {len(X_train)} samples on {self.device} "
            f"for {self.num_epochs} epochs..."
        )

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    logits = self.model(batch_x)
                    loss = criterion(logits, batch_y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item() * len(batch_x)

            if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == self.num_epochs - 1:
                avg_loss = epoch_loss / len(X_train)
                print(f"Epoch {epoch + 1}/{self.num_epochs} - loss={avg_loss:.4f}")

        val_proba = self._predict_probabilities(X_val)
        self._print_threshold_summary(y_val, val_proba)
        print("Training done.")

    def predict(self, test_X):
        """Predict multi-hot labels from thresholded class probabilities."""
        probabilities = self._predict_probabilities(test_X)
        return (probabilities >= self.thresholds).astype(int)
