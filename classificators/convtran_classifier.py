import copy
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from classificators.torch_multilabel_utils import (
    optimize_probability_thresholds,
    safe_mcc,
    sample_evenly,
)
from classificators.window_features import ensure_channel_first_windows, sample_training_windows


class TimeAbsolutePositionEncoding(nn.Module):
    """Length-aware sinusoidal encoding adapted for short time-series windows."""

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )
        scale = d_model / max_len
        pe[:, 0::2] = torch.sin((position * div_term) * scale)
        pe[:, 1::2] = torch.cos((position * div_term) * scale)
        self.register_buffer("pe", scale_factor * pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.shape[1]
        return self.dropout(x + self.pe[:, :seq_len])


class RelativeMultiHeadAttention(nn.Module):
    """Multi-head self-attention with per-head relative distance bias."""

    def __init__(self, d_model, num_heads, max_seq_len, dropout=0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_seq_len = max_seq_len

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * max_seq_len - 1), num_heads)
        )
        self.register_buffer(
            "relative_index",
            self._build_relative_index(max_seq_len),
            persistent=False,
        )
        nn.init.trunc_normal_(self.relative_bias_table, std=0.02)

    @staticmethod
    def _build_relative_index(seq_len):
        coords = torch.arange(seq_len)
        relative_coords = coords[:, None] - coords[None, :]
        return relative_coords + seq_len - 1

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds configured max_seq_len={self.max_seq_len}."
            )

        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        relative_bias = self.relative_bias_table[self.relative_index[:seq_len, :seq_len]]
        relative_bias = relative_bias.permute(2, 0, 1).unsqueeze(0)
        attn_scores = attn_scores + relative_bias

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.out_dropout(self.out_proj(context))


class ConvTranEncoderBlock(nn.Module):
    """Pre-norm transformer block with relative-bias attention."""

    def __init__(self, d_model, num_heads, ff_mult, max_seq_len, dropout=0.1):
        super().__init__()
        ff_dim = d_model * ff_mult
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RelativeMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class ConvTokenizer(nn.Module):
    """Two-layer temporal convolutional tokenizer."""

    def __init__(self, in_channels, d_model, kernel_sizes=(7, 5), dropout=0.1):
        super().__init__()
        if len(kernel_sizes) != 2:
            raise ValueError("conv_kernel_sizes must contain exactly two kernel sizes.")

        first_kernel, second_kernel = kernel_sizes
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=first_kernel, padding=first_kernel // 2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=second_kernel, padding=second_kernel // 2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.proj(x).transpose(1, 2)


class ConvTranBackbone(nn.Module):
    """ConvTran-inspired feature extractor with shared encoder for classification/pretraining."""

    def __init__(
        self,
        in_channels,
        d_model,
        num_heads,
        num_layers,
        ff_mult,
        seq_len,
        conv_kernel_sizes=(7, 5),
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.tokenizer = ConvTokenizer(
            in_channels=in_channels,
            d_model=d_model,
            kernel_sizes=conv_kernel_sizes,
            dropout=dropout,
        )
        self.position_encoding = TimeAbsolutePositionEncoding(
            d_model=d_model,
            dropout=dropout,
            max_len=seq_len,
        )
        self.encoder_blocks = nn.ModuleList(
            [
                ConvTranEncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_mult=ff_mult,
                    max_seq_len=seq_len,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    def tokenize(self, x):
        return self.tokenizer(x)

    def encode_tokens(self, tokens):
        x = self.position_encoding(tokens)
        for block in self.encoder_blocks:
            x = block(x)
        return self.final_norm(x)

    def forward(self, x):
        return self.encode_tokens(self.tokenize(x))


class ConvTranClassifierNet(nn.Module):
    """Backbone plus pooled multi-label classification head."""

    def __init__(
        self,
        in_channels,
        n_classes,
        seq_len,
        d_model=96,
        num_heads=4,
        num_layers=3,
        ff_mult=4,
        conv_kernel_sizes=(7, 5),
        dropout=0.1,
    ):
        super().__init__()
        self.backbone = ConvTranBackbone(
            in_channels=in_channels,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_mult=ff_mult,
            seq_len=seq_len,
            conv_kernel_sizes=conv_kernel_sizes,
            dropout=dropout,
        )
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        encoded = self.backbone(x)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


class ConvTranWindowClassifier:
    """ConvTran-inspired window classifier with optional masked-pretraining."""

    def __init__(
        self,
        classes,
        sensor_names,
        max_train_samples=120000,
        max_val_samples=50000,
        train_batch_size=512,
        inference_batch_size=2048,
        num_epochs=20,
        learning_rate=3e-4,
        weight_decay=1e-2,
        pos_weight_power=0.25,
        d_model=96,
        num_heads=4,
        num_layers=3,
        ff_mult=4,
        conv_kernel_sizes=(7, 5),
        dropout=0.1,
        use_pretraining=False,
        pretrain_epochs=12,
        mask_ratio=0.3,
        mask_span=8,
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
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_mult = ff_mult
        self.conv_kernel_sizes = conv_kernel_sizes
        self.dropout = dropout
        self.use_pretraining = use_pretraining
        self.pretrain_epochs = pretrain_epochs
        self.mask_ratio = mask_ratio
        self.mask_span = mask_span
        self.random_state = random_state
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = None
        self.reconstruction_head = None
        self.thresholds = np.full(len(classes), 0.5, dtype=np.float32)

    def _set_seed(self):
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _make_loader(self, X, y=None, shuffle=False):
        X = np.array(
            ensure_channel_first_windows(X, len(self.sensor_names)),
            dtype=np.float32,
            copy=True,
        )
        X_tensor = torch.from_numpy(X)

        if y is None:
            dataset = TensorDataset(X_tensor)
            batch_size = self.inference_batch_size
        else:
            y_tensor = torch.as_tensor(y, dtype=torch.float32)
            dataset = TensorDataset(X_tensor, y_tensor)
            batch_size = self.train_batch_size

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _build_model(self, seq_len):
        self.model = ConvTranClassifierNet(
            in_channels=len(self.sensor_names),
            n_classes=len(self.classes),
            seq_len=seq_len,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            ff_mult=self.ff_mult,
            conv_kernel_sizes=self.conv_kernel_sizes,
            dropout=self.dropout,
        ).to(self.device)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, len(self.sensor_names)),
        ).to(self.device)

    def _build_scheduler(self, optimizer, steps_per_epoch, num_epochs):
        total_steps = max(1, steps_per_epoch * num_epochs)
        warmup_steps = min(total_steps - 1, 2 * steps_per_epoch) if total_steps > 1 else 0

        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)

            if total_steps == warmup_steps:
                return 1.0

            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def _augment_batch(self, batch_x):
        noise = torch.randn_like(batch_x) * 0.01
        scale = torch.empty(
            batch_x.shape[0],
            batch_x.shape[1],
            1,
            device=batch_x.device,
        ).uniform_(0.95, 1.05)
        return batch_x * scale + noise

    def _make_span_mask(self, batch_size, seq_len, device):
        target_masked = max(1, int(round(seq_len * self.mask_ratio)))
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        max_start = max(1, seq_len - self.mask_span + 1)

        for row_idx in range(batch_size):
            masked = 0
            while masked < target_masked:
                start = torch.randint(0, max_start, (1,), device=device).item()
                end = min(seq_len, start + self.mask_span)
                before = mask[row_idx, start:end].sum().item()
                mask[row_idx, start:end] = True
                masked += (end - start) - before

        return mask

    def _compute_pretraining_loss(self, batch_x):
        tokens = self.model.backbone.tokenize(batch_x)
        seq_len = tokens.shape[1]
        mask = self._make_span_mask(tokens.shape[0], seq_len, batch_x.device)
        masked_tokens = tokens.clone()
        masked_count = int(mask.sum().item())
        masked_tokens[mask] = self.model.backbone.mask_token.view(1, -1).expand(masked_count, -1)

        encoded = self.model.backbone.encode_tokens(masked_tokens)
        reconstruction = self.reconstruction_head(encoded)
        targets = batch_x.transpose(1, 2)

        if mask.any():
            return torch.mean((reconstruction[mask] - targets[mask]) ** 2)
        return torch.mean((reconstruction - targets) ** 2)

    def _predict_probabilities(self, X):
        self.model.eval()
        loader = self._make_loader(X, y=None, shuffle=False)
        probabilities = []

        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                probabilities.append(torch.sigmoid(logits).cpu().numpy())

        return np.vstack(probabilities)

    def _evaluate_validation(self, y_true, y_proba):
        thresholds = optimize_probability_thresholds(y_true, y_proba)
        per_class_mcc = []

        for class_idx in range(y_true.shape[1]):
            predictions = (y_proba[:, class_idx] >= thresholds[class_idx]).astype(int)
            per_class_mcc.append(safe_mcc(y_true[:, class_idx], predictions))

        return thresholds, np.array(per_class_mcc, dtype=np.float32)

    def _run_pretraining(self, X_train):
        print(
            f"Pretraining ConvTran on {len(X_train)} windows on {self.device} "
            f"for {self.pretrain_epochs} epochs..."
        )
        loader = self._make_loader(X_train, y=np.zeros((len(X_train), 1), dtype=np.float32), shuffle=True)
        optimizer = torch.optim.AdamW(
            list(self.model.backbone.parameters()) + list(self.reconstruction_head.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = self._build_scheduler(optimizer, len(loader), self.pretrain_epochs)

        for epoch in range(self.pretrain_epochs):
            self.model.backbone.train()
            self.reconstruction_head.train()
            epoch_loss = 0.0

            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                loss = self._compute_pretraining_loss(batch_x)
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item() * len(batch_x)

            avg_loss = epoch_loss / len(loader.dataset)
            if epoch == 0 or (epoch + 1) % 4 == 0 or epoch == self.pretrain_epochs - 1:
                print(
                    f"Pretrain epoch {epoch + 1}/{self.pretrain_epochs} - "
                    f"loss={avg_loss:.4f}"
                )

    def train(self, train_data, val_data):
        self._set_seed()

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

        seq_len = ensure_channel_first_windows(X_train, len(self.sensor_names)).shape[2]
        self._build_model(seq_len=seq_len)

        if self.use_pretraining:
            self._run_pretraining(X_train)

        loader = self._make_loader(X_train, y_train, shuffle=True)
        y_train_tensor = torch.as_tensor(y_train, dtype=torch.float32)
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
        scheduler = self._build_scheduler(optimizer, len(loader), self.num_epochs)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_mcc = -np.inf
        best_state = copy.deepcopy(self.model.state_dict())
        best_thresholds = self.thresholds.copy()

        print(
            f"Training ConvTran with {len(X_train)} samples on {self.device} "
            f"for {self.num_epochs} epochs..."
        )

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_x = self._augment_batch(batch_x)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item() * len(batch_x)

            avg_loss = epoch_loss / len(loader.dataset)
            val_proba = self._predict_probabilities(X_val)
            thresholds, per_class_mcc = self._evaluate_validation(y_val, val_proba)
            mean_val_mcc = float(per_class_mcc.mean())

            print(
                f"Epoch {epoch + 1}/{self.num_epochs} - loss={avg_loss:.4f} "
                f"- val_mcc={mean_val_mcc:.4f}"
            )

            if mean_val_mcc > best_val_mcc:
                best_val_mcc = mean_val_mcc
                best_state = copy.deepcopy(self.model.state_dict())
                best_thresholds = thresholds.copy()

        self.model.load_state_dict(best_state)
        self.thresholds = best_thresholds

        print("Best validation thresholds:")
        for class_idx, class_name in enumerate(self.classes):
            print(f"  {class_name}: {self.thresholds[class_idx]:.4f}")
        print(f"Best validation mean MCC: {best_val_mcc:.4f}")
        print("Training done.")

    def predict(self, test_X):
        probabilities = self._predict_probabilities(test_X)
        return (probabilities >= self.thresholds).astype(int)
