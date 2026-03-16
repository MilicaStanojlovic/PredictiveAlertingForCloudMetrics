"""
Feature engineering for sliding-window incident prediction.

Design rationale:
- Raw metric values alone are weak predictors; derived features capture dynamics
- Rate-of-change detects sudden spikes missed by absolute thresholds
- Rolling statistics encode short-term volatility
- Cross-metric features (e.g. cpu × latency) capture correlated degradation
- All features are computed inside the window → no future leakage
"""

import numpy as np
from typing import List, Optional


FEATURE_COLS = ["cpu", "latency_ms", "error_rate", "request_rate"]


def engineer_features(X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
    """
    Transform raw (n_samples, W, n_raw_features) windows into
    flat feature vectors (n_samples, n_engineered_features).

    Features per metric:
        - mean, std, min, max, last value
        - linear trend (slope via least-squares)
        - rate of change: (last - first) / W
        - rolling max of 1-step differences (spike detector)
        - percentile 90 (heavy tail sensitivity)
    Cross-metric:
        - cpu * latency_ms (combined pressure)
        - error_rate * request_rate (error volume)
    """
    if feature_names is None:
        feature_names = FEATURE_COLS

    n, W, F = X.shape
    t = np.arange(W, dtype=np.float32)
    t_centered = t - t.mean()

    feature_vectors = []

    for f_idx in range(F):
        col = X[:, :, f_idx]  # (n, W)

        mean_ = col.mean(axis=1)
        std_ = col.std(axis=1) + 1e-8
        min_ = col.min(axis=1)
        max_ = col.max(axis=1)
        last_ = col[:, -1]
        first_ = col[:, 0]

        # Linear trend slope
        denom = (t_centered ** 2).sum()
        slope_ = ((col - mean_[:, None]) * t_centered[None, :]).sum(axis=1) / denom

        # Rate of change
        roc_ = (last_ - first_) / (W + 1e-8)

        # Max 1-step diff (spike magnitude)
        diffs = np.abs(np.diff(col, axis=1))  # (n, W-1)
        max_diff_ = diffs.max(axis=1)

        # 90th percentile
        p90_ = np.percentile(col, 90, axis=1)

        feature_vectors += [mean_, std_, min_, max_, last_, slope_, roc_, max_diff_, p90_]

    # Cross-metric features (indices into feature_names)
    if "cpu" in feature_names and "latency_ms" in feature_names:
        cpu_idx = feature_names.index("cpu")
        lat_idx = feature_names.index("latency_ms")
        cpu_mean = X[:, :, cpu_idx].mean(axis=1)
        lat_mean = X[:, :, lat_idx].mean(axis=1)
        feature_vectors.append(cpu_mean * lat_mean)

    if "error_rate" in feature_names and "request_rate" in feature_names:
        err_idx = feature_names.index("error_rate")
        req_idx = feature_names.index("request_rate")
        err_mean = X[:, :, err_idx].mean(axis=1)
        req_mean = X[:, :, req_idx].mean(axis=1)
        feature_vectors.append(err_mean * req_mean)

    return np.column_stack(feature_vectors).astype(np.float32)


def get_feature_names(feature_names: Optional[List[str]] = None) -> List[str]:
    """Return names corresponding to engineer_features output."""
    if feature_names is None:
        feature_names = FEATURE_COLS

    suffixes = ["mean", "std", "min", "max", "last", "slope", "roc", "max_diff", "p90"]
    names = [f"{col}_{s}" for col in feature_names for s in suffixes]

    if "cpu" in feature_names and "latency_ms" in feature_names:
        names.append("cpu_x_latency")
    if "error_rate" in feature_names and "request_rate" in feature_names:
        names.append("error_x_requests")

    return names
