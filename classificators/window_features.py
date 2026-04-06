import numpy as np


def ensure_channel_first_windows(windows, n_sensors):
    """Normalize input windows to shape ``(n_samples, n_sensors, seq_len)``."""
    if windows.ndim != 3:
        raise ValueError(f"Expected 3D window tensor, got shape {windows.shape}.")

    if windows.shape[1] == n_sensors:
        return windows

    if windows.shape[2] == n_sensors:
        return np.transpose(windows, (0, 2, 1))

    raise ValueError(
        "Could not infer sensor axis for windows with "
        f"shape {windows.shape} and n_sensors={n_sensors}."
    )


def sample_training_windows(X, y, max_samples, random_state):
    """Return a deterministic subset of training windows to limit tree-model cost."""
    if len(X) <= max_samples:
        return X, y

    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(len(X), size=max_samples, replace=False)
    return X[sample_idx], y[sample_idx]


def _zero_crossing_rate(windows):
    """Compute the fraction of sign changes per channel within each window."""
    centered = windows - np.mean(windows, axis=2, keepdims=True)
    signs = centered >= 0
    return np.mean(signs[:, :, 1:] != signs[:, :, :-1], axis=2)


def _magnitude_summary(windows, channel_indices):
    """Summarize vector magnitude for a sensor group such as accel or gyro axes."""
    magnitudes = np.linalg.norm(windows[:, channel_indices, :], axis=1)
    return np.column_stack(
        [
            np.mean(magnitudes, axis=1),
            np.std(magnitudes, axis=1),
            np.max(magnitudes, axis=1),
        ]
    )


def _safe_std(values, axis, keepdims=False):
    """Compute a standard deviation with a floor to avoid divide-by-zero."""
    std = np.std(values, axis=axis, keepdims=keepdims)
    return np.where(std < 1e-6, 1.0, std)


def _skewness(windows):
    """Estimate per-channel skewness for each window."""
    centered = windows - np.mean(windows, axis=2, keepdims=True)
    std = _safe_std(centered, axis=2, keepdims=True)
    normalized = centered / std
    return np.mean(normalized ** 3, axis=2)


def _kurtosis(windows):
    """Estimate excess kurtosis per channel for each window."""
    centered = windows - np.mean(windows, axis=2, keepdims=True)
    std = _safe_std(centered, axis=2, keepdims=True)
    normalized = centered / std
    return np.mean(normalized ** 4, axis=2) - 3.0


def _autocorrelation(windows, lag):
    """Compute lagged autocorrelation per channel for each window."""
    if lag >= windows.shape[2]:
        return np.zeros((windows.shape[0], windows.shape[1]), dtype=np.float32)

    x0 = windows[:, :, :-lag]
    x1 = windows[:, :, lag:]
    x0_centered = x0 - np.mean(x0, axis=2, keepdims=True)
    x1_centered = x1 - np.mean(x1, axis=2, keepdims=True)

    numerator = np.mean(x0_centered * x1_centered, axis=2)
    denominator = _safe_std(x0_centered, axis=2) * _safe_std(x1_centered, axis=2)
    return numerator / denominator


def _spectral_features(windows):
    """Extract simple FFT-derived features: dominant bin and spectral entropy."""
    spectrum = np.fft.rfft(windows, axis=2)
    power = np.abs(spectrum) ** 2
    non_dc_power = power[:, :, 1:]

    dominant_frequency = np.argmax(non_dc_power, axis=2).astype(np.float32)
    power_sum = np.sum(non_dc_power, axis=2, keepdims=True)
    normalized_power = non_dc_power / np.where(power_sum < 1e-6, 1.0, power_sum)
    spectral_entropy = -np.sum(
        normalized_power * np.log(normalized_power + 1e-12),
        axis=2,
    )

    return dominant_frequency, spectral_entropy


def _pairwise_correlations(windows):
    """Compute pairwise per-window correlations across sensor channels."""
    n_samples, n_channels, _ = windows.shape
    centered = windows - np.mean(windows, axis=2, keepdims=True)
    normalized = centered / _safe_std(centered, axis=2, keepdims=True)
    features = []

    for left in range(n_channels):
        for right in range(left + 1, n_channels):
            corr = np.mean(normalized[:, left, :] * normalized[:, right, :], axis=1)
            features.append(corr)

    if not features:
        return np.zeros((n_samples, 0), dtype=np.float32)

    return np.column_stack(features)


def extract_window_features(windows, sensor_names):
    """
    Convert time-series windows into tabular features for tree-based classifiers.

    The resulting feature matrix combines per-channel summary statistics,
    temporal-shape features, FFT summaries, vector-magnitude summaries, and
    cross-channel correlations.
    """
    windows = ensure_channel_first_windows(windows, len(sensor_names)).astype(np.float32, copy=False)

    q10 = np.percentile(windows, 10, axis=2)
    q25 = np.percentile(windows, 25, axis=2)
    q75 = np.percentile(windows, 75, axis=2)
    q90 = np.percentile(windows, 90, axis=2)
    dominant_frequency, spectral_entropy = _spectral_features(windows)

    feature_blocks = [
        np.mean(windows, axis=2),
        np.std(windows, axis=2),
        np.min(windows, axis=2),
        np.max(windows, axis=2),
        np.median(windows, axis=2),
        q10,
        q25,
        q75,
        q90,
        q75 - q25,
        np.sqrt(np.mean(np.square(windows), axis=2)),
        _zero_crossing_rate(windows),
        _skewness(windows),
        _kurtosis(windows),
        _autocorrelation(windows, lag=1),
        _autocorrelation(windows, lag=5),
        _autocorrelation(windows, lag=10),
        dominant_frequency,
        spectral_entropy,
    ]

    sensor_index = {name: idx for idx, name in enumerate(sensor_names)}
    accel_indices = [sensor_index[name] for name in ("Acc.x", "Acc.y", "Acc.z") if name in sensor_index]
    gyro_indices = [sensor_index[name] for name in ("Gyro.x", "Gyro.y", "Gyro.z") if name in sensor_index]

    if len(accel_indices) == 3:
        feature_blocks.append(_magnitude_summary(windows, accel_indices))
    if len(gyro_indices) == 3:
        feature_blocks.append(_magnitude_summary(windows, gyro_indices))

    feature_blocks.append(_pairwise_correlations(windows))

    return np.hstack(feature_blocks).astype(np.float32, copy=False)
