from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from classificators.window_features import ensure_channel_first_windows


@dataclass
class RocketKernelGroup:
    """A batch of ROCKET kernels that share convolution hyperparameters."""

    length: int
    dilation: int
    padding: int
    weights: np.ndarray
    bias: np.ndarray


class RocketTransform:
    """GPU-capable ROCKET-style random convolutional feature transform."""

    def __init__(
        self,
        num_kernels=256,
        kernel_lengths=(7, 9, 11),
        max_channels_per_kernel=3,
        random_state=42,
    ):
        self.num_kernels = num_kernels
        self.kernel_lengths = tuple(kernel_lengths)
        self.max_channels_per_kernel = max_channels_per_kernel
        self.random_state = random_state
        self.groups = []
        self.n_sensors = None
        self.seq_len = None
        self._device_cache = {}

    def fit(self, windows):
        """Sample and group random kernels for the observed input shape."""
        windows = np.asarray(windows)
        if windows.ndim != 3:
            raise ValueError(f"Expected 3D windows, got shape {windows.shape}.")

        self.n_sensors = windows.shape[1]
        self.seq_len = windows.shape[2]

        rng = np.random.default_rng(self.random_state)
        grouped_kernels = {}

        for _ in range(self.num_kernels):
            length, dilation, padding, dense_weight, bias = self._sample_kernel(rng)
            grouped_kernels.setdefault((length, dilation, padding), []).append((dense_weight, bias))

        self.groups = []
        for (length, dilation, padding), kernels in grouped_kernels.items():
            weights = np.stack([weight for weight, _ in kernels]).astype(np.float32, copy=False)
            bias = np.array([kernel_bias for _, kernel_bias in kernels], dtype=np.float32)
            self.groups.append(
                RocketKernelGroup(
                    length=length,
                    dilation=dilation,
                    padding=padding,
                    weights=weights,
                    bias=bias,
                )
            )

        self._device_cache = {}
        return self

    def _sample_kernel(self, rng):
        """Create one sparse multivariate random kernel."""
        length = int(rng.choice(self.kernel_lengths))
        num_channels = int(rng.integers(1, min(self.n_sensors, self.max_channels_per_kernel) + 1))
        channel_indices = np.sort(rng.choice(self.n_sensors, size=num_channels, replace=False))

        max_dilation = max(1, (self.seq_len - 1) // max(1, length - 1))
        dilation = int(rng.integers(1, max_dilation + 1))
        effective_length = (length - 1) * dilation + 1
        padding = effective_length // 2 if rng.random() < 0.5 else 0

        kernel_weights = rng.normal(size=(num_channels, length)).astype(np.float32)
        kernel_weights -= np.mean(kernel_weights, axis=1, keepdims=True)
        kernel_weights /= np.sum(np.abs(kernel_weights), axis=1, keepdims=True) + 1e-6

        dense_weight = np.zeros((self.n_sensors, length), dtype=np.float32)
        dense_weight[channel_indices] = kernel_weights

        bias = float(rng.uniform(-1.0, 1.0))
        return length, dilation, padding, dense_weight, bias

    def _get_device_groups(self, device):
        """Materialize grouped kernels on the target torch device."""
        device_key = str(device)
        if device_key not in self._device_cache:
            self._device_cache[device_key] = [
                {
                    "weights": torch.as_tensor(group.weights, dtype=torch.float32, device=device),
                    "bias": torch.as_tensor(group.bias, dtype=torch.float32, device=device),
                }
                for group in self.groups
            ]
        return self._device_cache[device_key]

    def transform(self, windows, device):
        """Convert windows into concatenated max and PPV ROCKET features."""
        if not self.groups:
            raise RuntimeError("RocketTransform must be fit before calling transform().")

        windows = np.array(
            ensure_channel_first_windows(windows, self.n_sensors),
            dtype=np.float32,
            copy=True,
        )
        x = torch.as_tensor(windows, dtype=torch.float32, device=device)

        feature_blocks = []
        for group, device_group in zip(self.groups, self._get_device_groups(device), strict=True):
            responses = F.conv1d(
                x,
                device_group["weights"],
                bias=device_group["bias"],
                stride=1,
                padding=group.padding,
                dilation=group.dilation,
            )
            max_response = responses.amax(dim=2)
            positive_proportion = (responses > 0.0).float().mean(dim=2)
            feature_blocks.append(torch.stack((max_response, positive_proportion), dim=2).flatten(start_dim=1))

        return torch.cat(feature_blocks, dim=1)
