"""
Main training and evaluation pipeline.

Run:
    python train.py

This script:
  1. Generates synthetic CloudWatch-style metrics with incident labels
  2. Creates sliding-window supervised dataset
  3. Splits chronologically (no shuffle — respects temporal order)
  4. Engineers features from each window
  5. Trains GBT classifier
  6. Selects alert threshold targeting 80% incident recall
  7. Reports window-level and incident-level metrics
  8. Saves model + metrics to ./models/
"""

import numpy as np
import json
from pathlib import Path

from src.data_generator import generate_cloud_metrics, create_sliding_window_dataset, FEATURE_COLS
from src.features import engineer_features, get_feature_names
from src.model import train, evaluate, save_metrics
from src.incident_eval import evaluate_incidents, print_incident_report


# ── Hyperparameters ────────────────────────────────────────────────────────────
W = 30          # lookback window: 30 minutes of history
H = 10          # prediction horizon: predict incidents in next 10 minutes
STRIDE = 1      # window stride (1 = fully overlapping)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST = remaining 15%

N_STEPS = 50_000
TARGET_RECALL = 0.80
MODEL_DIR = Path("models")
# ──────────────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("Incident Prediction Pipeline")
    print("=" * 60)

    # 1. Generate data
    print(f"\n[1/6] Generating {N_STEPS} steps of synthetic metrics...")
    df, incidents = generate_cloud_metrics(n_steps=N_STEPS)
    print(f"      {len(incidents)} incidents | prevalence={df['incident'].mean():.2%}")

    # 2. Create sliding window dataset
    print(f"\n[2/6] Creating sliding windows (W={W}, H={H}, stride={STRIDE})...")
    X_raw, y = create_sliding_window_dataset(df, W=W, H=H, stride=STRIDE)
    print(f"      X shape: {X_raw.shape} | positive rate: {y.mean():.2%}")

    # 3. Chronological split — CRITICAL: never shuffle time series
    n = len(y)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train_raw, y_train = X_raw[:train_end],     y[:train_end]
    X_val_raw,   y_val   = X_raw[train_end:val_end], y[train_end:val_end]
    X_test_raw,  y_test  = X_raw[val_end:],        y[val_end:]

    print(f"\n[3/6] Chronological split:")
    print(f"      Train : {len(y_train):>6} samples ({y_train.mean():.2%} positive)")
    print(f"      Val   : {len(y_val):>6} samples ({y_val.mean():.2%} positive)")
    print(f"      Test  : {len(y_test):>6} samples ({y_test.mean():.2%} positive)")

    # 4. Feature engineering
    print("\n[4/6] Engineering features...")
    feature_names = get_feature_names(FEATURE_COLS)
    X_train = engineer_features(X_train_raw)
    X_val   = engineer_features(X_val_raw)
    X_test  = engineer_features(X_test_raw)
    print(f"      {X_train.shape[1]} features per window")

    # 5. Train
    print("\n[5/6] Training GradientBoostingClassifier...")
    model = train(
        X_train, y_train,
        model_dir=MODEL_DIR,
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
    )

    # Validation check
    print("\n--- Validation set ---")
    val_metrics = evaluate(model, X_val, y_val, target_recall=TARGET_RECALL)
    threshold = val_metrics["threshold"]

    # 6. Test evaluation
    print("\n[6/6] Test set evaluation (held-out)")
    print("-" * 40)
    test_metrics = evaluate(model, X_test, y_test, threshold=threshold)

    # Incident-level evaluation on test set
    # Map test windows back to global time indices
    test_start_idx = val_end  # first window index in test set (global)
    # Incidents fully in the test period
    test_incidents = [
        inc for inc in incidents
        if inc.start >= test_start_idx + W and inc.end < n + W + H
    ]
    # Shift incident indices to be relative to test window space
    # Window index i in test set corresponds to global time i + test_start_idx

    class ShiftedIncident:
        def __init__(self, start, end, severity):
            self.start = start - test_start_idx
            self.end   = end   - test_start_idx
            self.severity = severity

    shifted_incidents = [
        ShiftedIncident(inc.start, inc.end, inc.severity)
        for inc in test_incidents
    ]

    y_prob_test = test_metrics["y_prob"]
    incident_results, incident_summary = evaluate_incidents(
        y_prob=y_prob_test,
        incidents=shifted_incidents,
        W=W,
        H=H,
        threshold=threshold,
    )
    print_incident_report(incident_results, incident_summary)

    # Save everything
    MODEL_DIR.mkdir(exist_ok=True)
    combined = {
        "hyperparameters": {"W": W, "H": H, "stride": STRIDE, "target_recall": TARGET_RECALL},
        "window_metrics_val": {k: v for k, v in val_metrics.items() if not hasattr(v, '__len__') or k == 'threshold'},
        "window_metrics_test": {k: v for k, v in test_metrics.items() if k not in ("y_prob", "y_pred")},
        "incident_metrics_test": incident_summary,
    }
    save_metrics(combined, MODEL_DIR / "metrics.json")

    # Feature importances
    clf = model.named_steps["clf"]
    importances = clf.feature_importances_
    top_features = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:15]
    print("\n=== Top 15 Feature Importances ===")
    for name, imp in top_features:
        bar = "█" * int(imp * 200)
        print(f"  {name:<35} {imp:.4f} {bar}")

    print(f"\nModel + metrics saved to {MODEL_DIR}/")
    return model, test_metrics, incident_summary


if __name__ == "__main__":
    main()
