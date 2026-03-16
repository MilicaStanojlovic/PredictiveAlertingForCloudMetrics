"""
Model training, evaluation, and alert threshold selection.

Model choice: Gradient Boosted Trees (XGBoost / LightGBM-style via sklearn GBT)
-----------------------------------------------------------------------
Rationale:
1. Tabular features from sliding windows — tree ensembles consistently outperform
   linear models and are competitive with LSTMs on tabular data.
2. Robust to heavy-tailed feature distributions (no normality assumption).
3. Native handling of class imbalance via scale_pos_weight.
4. Fast inference (~1 ms per prediction) suitable for per-minute Lambda execution.
5. Model artifacts are small (<10 MB) → easy S3 storage and Lambda deployment.
6. Interpretable feature importances aid post-incident analysis.

Alternative considered: LSTM / Transformer
- Better at capturing long temporal dependencies in raw sequences.
- Requires more data and tuning; slower inference; larger artifacts.
- Preferred when W >> 60 or when raw sequence patterns dominate.

Alternative considered: Isolation Forest / anomaly detection
- Unsupervised, no labels needed.
- Lower recall for incident prediction (detects anomalies, not future incidents).
"""

import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight


def build_model(
    class_weight: str = "balanced",
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    random_state: int = 42,
) -> Pipeline:
    """
    Build a sklearn Pipeline: StandardScaler → GradientBoostingClassifier.

    StandardScaler is included for numerical stability even though GBT
    doesn't strictly require it — makes the pipeline drop-in compatible
    with linear/neural alternatives.
    """
    clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        min_samples_leaf=20,
        max_features="sqrt",
        random_state=random_state,
        validation_fraction=0.1,
        n_iter_no_change=20,
        tol=1e-4,
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])
    return pipeline


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_dir: Optional[Path] = None,
    **model_kwargs,
) -> Pipeline:
    """Train model, optionally save to disk."""
    model = build_model(**model_kwargs)

    # GradientBoostingClassifier uses sample_weight for imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_map = dict(zip(classes, weights))
    sample_weights = np.array([weight_map[c] for c in y_train])

    model.fit(X_train, y_train, clf__sample_weight=sample_weights)

    if model_dir:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / "model.joblib")
        print(f"Model saved to {model_dir / 'model.joblib'}")

    return model


def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.80,
) -> Tuple[float, Dict]:
    """
    Select decision threshold that achieves target_recall while
    maximising precision. Returns threshold and metrics at that point.

    Alert system design: we prefer recall over precision because:
    - Missing an incident (false negative) is costly (outage)
    - False alarms are annoying but tolerable if not too frequent
    - Target recall ≈ 0.80 matches the task specification
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # Find threshold achieving >= target_recall with highest precision
    valid = recalls[:-1] >= target_recall  # last element has no threshold
    if valid.any():
        idx = np.where(valid)[0]
        # Among valid, pick highest precision
        best_idx = idx[np.argmax(precisions[:-1][idx])]
        threshold = float(thresholds[best_idx])
    else:
        # Fallback: closest recall to target
        idx = np.argmin(np.abs(recalls[:-1] - target_recall))
        threshold = float(thresholds[idx])

    # Compute metrics at selected threshold
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    metrics = {
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "avg_precision": round(average_precision_score(y_true, y_prob), 4),
    }
    return threshold, metrics


def evaluate(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: Optional[float] = None,
    target_recall: float = 0.80,
) -> Dict:
    """Full evaluation: AUC, PR curve, threshold selection, classification report."""
    y_prob = model.predict_proba(X_test)[:, 1]

    selected_threshold, metrics = select_threshold(y_test, y_prob, target_recall)
    if threshold is None:
        threshold = selected_threshold

    y_pred = (y_prob >= threshold).astype(int)
    print("\n=== Evaluation Results ===")
    print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(f"Avg Precision:     {metrics['avg_precision']:.4f}")
    print(f"Selected threshold:{metrics['threshold']:.4f}")
    print(f"Precision:         {metrics['precision']:.4f}")
    print(f"Recall:            {metrics['recall']:.4f}")
    print(f"F1:                {metrics['f1']:.4f}")
    print(f"FPR:               {metrics['fpr']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={metrics['tn']}  FP={metrics['fp']}")
    print(f"  FN={metrics['fn']}  TP={metrics['tp']}")

    metrics["y_prob"] = y_prob
    metrics["y_pred"] = y_pred
    return metrics


def save_metrics(metrics: Dict, path: Path):
    """Save serializable metrics (excluding arrays) to JSON."""
    serializable = {k: v for k, v in metrics.items()
                    if not isinstance(v, np.ndarray)}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
