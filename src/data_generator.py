"""
Synthetic CloudWatch-style time-series generator with labeled incident intervals.

Design rationale:
- Simulates common cloud metric patterns: CPU, latency, error rate, request count
- Incidents are triggered by cascading anomalies (spike → latency → errors)
- Heavy-tailed noise (Student-t) reflects real cloud metric distributions
- Non-stationary: regime shifts model traffic surges or config changes
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional

FEATURE_COLS = ["cpu", "latency_ms", "error_rate", "request_rate"]


@dataclass
class IncidentInterval:
    start: int
    end: int
    severity: float  # 0-1


def generate_cloud_metrics(
    n_steps: int = 50_000,
    sampling_interval_sec: int = 60,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[IncidentInterval]]:
    """
    Generate synthetic multi-variate cloud metrics with incident labels.

    Returns:
        df: DataFrame with columns [timestamp, cpu, latency_ms, error_rate, request_rate]
        incidents: list of IncidentInterval objects
    """
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(
        start="2024-01-01", periods=n_steps, freq=f"{sampling_interval_sec}s"
    )

    # --- Base signal components ---
    t = np.arange(n_steps)

    # Diurnal pattern (24h cycle)
    diurnal = 0.3 * np.sin(2 * np.pi * t / (86400 / sampling_interval_sec))

    # Weekly pattern
    weekly = 0.1 * np.sin(2 * np.pi * t / (7 * 86400 / sampling_interval_sec))

    # Slow trend (simulates gradual load growth)
    trend = 0.0001 * t / n_steps

    base_load = 0.4 + diurnal + weekly + trend  # CPU baseline ~40%

    # Heavy-tailed noise (df=4 Student-t is realistic for cloud metrics)
    noise_cpu = rng.standard_t(df=4, size=n_steps) * 0.05
    noise_lat = rng.standard_t(df=4, size=n_steps) * 10.0
    noise_err = np.abs(rng.standard_t(df=4, size=n_steps)) * 0.002

    # --- Regime shifts (non-stationarity) ---
    regime = np.ones(n_steps)
    n_shifts = 5
    shift_points = rng.integers(n_steps // 10, n_steps - n_steps // 10, size=n_shifts)
    for sp in shift_points:
        duration = rng.integers(500, 3000)
        end = min(sp + duration, n_steps)
        regime[sp:end] *= rng.uniform(1.2, 1.8)

    # --- CPU utilization ---
    cpu = np.clip(base_load * regime + noise_cpu, 0.05, 1.0)

    # --- Request rate (req/s) --- correlated with diurnal
    request_rate = np.clip(
        100 * (0.5 + diurnal + weekly) * regime
        + rng.standard_t(df=6, size=n_steps) * 5,
        1,
        500,
    )

    # --- Latency (ms) --- increases with CPU pressure
    latency_ms = np.clip(
        50 + 200 * cpu**2 * regime + noise_lat, 10, 5000
    )

    # --- Error rate --- near-zero baseline, spikes during incidents
    error_rate = np.clip(0.001 + np.abs(noise_err), 0, 0.05)

    # --- Generate incidents ---
    incidents: List[IncidentInterval] = []
    incident_mask = np.zeros(n_steps, dtype=bool)

    # Space incidents out: roughly 1 per 2000 steps, random gaps
    n_incidents = n_steps // 2000
    min_gap = 500  # minimum steps between incidents

    candidate_starts = []
    current = rng.integers(200, 600)
    while current < n_steps - 300:
        candidate_starts.append(current)
        current += rng.integers(min_gap, min_gap * 4)

    # Keep a random subset
    chosen = rng.choice(
        candidate_starts,
        size=min(n_incidents, len(candidate_starts)),
        replace=False,
    )

    for start in sorted(chosen):
        severity = rng.uniform(0.4, 1.0)
        duration = int(rng.integers(30, 180))  # 30-180 min at 1-min resolution
        end = min(start + duration, n_steps - 1)

        incidents.append(IncidentInterval(start=int(start), end=int(end), severity=severity))
        incident_mask[start:end] = True

        # Inject anomalous signal into metrics during incident
        pre_cursor = max(0, start - rng.integers(5, 20))  # anomaly starts BEFORE incident

        # CPU spike
        cpu[pre_cursor:end] += severity * rng.uniform(0.2, 0.5)
        # Latency spike
        latency_ms[pre_cursor:end] += severity * rng.uniform(200, 800)
        # Error surge
        error_rate[start:end] += severity * rng.uniform(0.05, 0.3)
        # Request rate can drop (timeout cascade) or spike
        if rng.random() > 0.5:
            request_rate[start:end] *= rng.uniform(0.3, 0.7)
        else:
            request_rate[pre_cursor:end] *= rng.uniform(1.5, 3.0)

    # Clip after injection
    cpu = np.clip(cpu, 0.0, 1.0)
    latency_ms = np.clip(latency_ms, 5.0, 10000.0)
    error_rate = np.clip(error_rate, 0.0, 1.0)
    request_rate = np.clip(request_rate, 0.0, 1000.0)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "cpu": cpu,
            "latency_ms": latency_ms,
            "error_rate": error_rate,
            "request_rate": request_rate,
            "incident": incident_mask.astype(int),
        }
    )

    return df, incidents


def create_sliding_window_dataset(
    df: pd.DataFrame,
    W: int = 30,
    H: int = 10,
    feature_cols: Optional[List[str]] = None,
    label_col: str = "incident",
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert time-series DataFrame into supervised learning dataset.

    Window formulation:
        X[i] = metrics[i : i+W]          shape (W, n_features)
        y[i] = 1 if any incident in [i+W : i+W+H] else 0

    Args:
        W: lookback window (number of past steps)
        H: prediction horizon (number of future steps to check for incident)
        stride: step between windows (1 = maximum overlap)

    Returns:
        X: (n_samples, W, n_features)
        y: (n_samples,) binary labels
    """
    if feature_cols is None:
        feature_cols = ["cpu", "latency_ms", "error_rate", "request_rate"]

    features = df[feature_cols].values
    labels = df[label_col].values

    X_list, y_list = [], []
    n = len(df)

    for i in range(0, n - W - H + 1, stride):
        window = features[i : i + W]
        future_labels = labels[i + W : i + W + H]
        target = int(future_labels.any())
        X_list.append(window)
        y_list.append(target)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


if __name__ == "__main__":
    df, incidents = generate_cloud_metrics(n_steps=50_000)
    print(f"Generated {len(df)} time steps, {len(incidents)} incidents")
    print(f"Incident prevalence: {df['incident'].mean():.3%}")
    print(df.head())

    X, y = create_sliding_window_dataset(df, W=30, H=10)
    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
    print(f"Positive class ratio: {y.mean():.3%}")
