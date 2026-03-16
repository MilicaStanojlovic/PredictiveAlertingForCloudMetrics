"""
AWS Lambda: Periodic model retraining (runs daily via EventBridge).

Architecture:
  - Triggered by EventBridge Scheduler (cron: 0 2 * * ? *)
  - Fetches last 30 days of CloudWatch metrics via boto3
  - Retrains model and saves artifact to S3
  - Writes metrics to CloudWatch for monitoring retraining health

Environment variables:
  METRICS_BUCKET      S3 bucket for model artifacts and cached data
  MODEL_KEY_PREFIX    S3 key prefix, e.g. "incident-predictor/models/"
  DATA_KEY_PREFIX     S3 key prefix for cached metric CSVs
  CLOUDWATCH_NS       CloudWatch namespace for training metrics
  METRIC_NAMES        Comma-separated list of CloudWatch metric names to fetch
  AWS_REGION          AWS region

Usage:
  Deploy as a Lambda function with a 15-minute timeout and 512 MB memory.
  Layer or container must include: scikit-learn, numpy, pandas, joblib.
"""

import json
import os
import io
import logging
from datetime import datetime, timedelta, timezone

import boto3
import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── Inline imports (same code as src/ but packaged with Lambda layer) ─────────
# In a real deployment, these would come from a Lambda layer or container image.
# Shown here for completeness.
from src.data_generator import create_sliding_window_dataset, FEATURE_COLS
from src.features import engineer_features
from src.model import train, select_threshold, save_metrics
# ──────────────────────────────────────────────────────────────────────────────

S3 = boto3.client("s3")
CW = boto3.client("cloudwatch")

BUCKET         = os.environ["METRICS_BUCKET"]
MODEL_PREFIX   = os.environ.get("MODEL_KEY_PREFIX", "incident-predictor/models/")
DATA_PREFIX    = os.environ.get("DATA_KEY_PREFIX", "incident-predictor/data/")
CW_NAMESPACE   = os.environ.get("CLOUDWATCH_NS", "IncidentPredictor")
W              = int(os.environ.get("WINDOW_W", "30"))
H              = int(os.environ.get("HORIZON_H", "10"))
TARGET_RECALL  = float(os.environ.get("TARGET_RECALL", "0.80"))
LOOKBACK_DAYS  = int(os.environ.get("LOOKBACK_DAYS", "30"))


def fetch_cloudwatch_metrics(metric_names: list, namespace: str, dimensions: list,
                              lookback_days: int = 30) -> pd.DataFrame:
    """
    Fetch CloudWatch metrics and return as a DataFrame.
    Period = 60s (1 minute resolution).
    """
    cw = boto3.client("cloudwatch")
    end_time   = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=lookback_days)

    metric_data_queries = []
    for i, name in enumerate(metric_names):
        metric_data_queries.append({
            "Id": f"m{i}",
            "MetricStat": {
                "Metric": {
                    "Namespace": namespace,
                    "MetricName": name,
                    "Dimensions": dimensions,
                },
                "Period": 60,
                "Stat": "Average",
            },
            "ReturnData": True,
        })

    response = cw.get_metric_data(
        MetricDataQueries=metric_data_queries,
        StartTime=start_time,
        EndTime=end_time,
        ScanBy="TimestampAscending",
    )

    # Build DataFrame from results
    dfs = {}
    for result in response["MetricDataResults"]:
        idx = int(result["Id"][1:])
        name = metric_names[idx]
        dfs[name] = pd.Series(result["Values"], index=result["Timestamps"]).sort_index()

    df = pd.DataFrame(dfs)
    df.index.name = "timestamp"
    df = df.resample("1min").mean().ffill()  # fill gaps with forward fill
    return df.reset_index()


def load_incident_labels_from_s3(bucket: str, key: str) -> pd.Series:
    """
    Load incident label time-series from S3 (CSV with timestamp, incident columns).
    Labels are derived from existing alert conditions evaluated historically.
    """
    obj = S3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()), parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df["incident"]


def save_model_to_s3(model, threshold: float, metrics: dict, version: str):
    """Serialize and upload model + metadata to S3."""
    buf = io.BytesIO()
    joblib.dump({"model": model, "threshold": threshold, "version": version}, buf)
    buf.seek(0)
    key = f"{MODEL_PREFIX}{version}/model.joblib"
    S3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
    logger.info(f"Model saved to s3://{BUCKET}/{key}")

    # Also write as "latest" for the inference Lambda to always find
    latest_key = f"{MODEL_PREFIX}latest/model.joblib"
    buf.seek(0)
    S3.put_object(Bucket=BUCKET, Key=latest_key, Body=buf.getvalue())

    # Save metrics JSON
    metrics_key = f"{MODEL_PREFIX}{version}/metrics.json"
    S3.put_object(
        Bucket=BUCKET,
        Key=metrics_key,
        Body=json.dumps(metrics).encode(),
    )


def publish_training_metrics(metrics: dict, version: str):
    """Push retraining metrics to CloudWatch for dashboarding."""
    metric_data = []
    for k in ["roc_auc", "avg_precision", "recall", "precision", "fpr"]:
        if k in metrics:
            metric_data.append({
                "MetricName": f"Train_{k}",
                "Value": metrics[k],
                "Unit": "None",
                "Dimensions": [{"Name": "Version", "Value": version}],
            })
    if metric_data:
        CW.put_metric_data(Namespace=CW_NAMESPACE, MetricData=metric_data)


def lambda_handler(event, context):
    """Entry point for the retraining Lambda."""
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting retraining job version={version}")

    try:
        # 1. Fetch metrics
        metric_names = os.environ.get(
            "METRIC_NAMES", "CPUUtilization,Latency,ErrorRate,RequestCount"
        ).split(",")
        namespace = os.environ.get("SERVICE_NAMESPACE", "AWS/ApplicationELB")
        dimensions = json.loads(os.environ.get("DIMENSIONS", "[]"))

        df_metrics = fetch_cloudwatch_metrics(metric_names, namespace, dimensions, LOOKBACK_DAYS)
        logger.info(f"Fetched {len(df_metrics)} metric rows")

        # 2. Load incident labels (derived from existing alert conditions stored in S3)
        labels_key = os.environ.get("LABELS_KEY", f"{DATA_PREFIX}incident_labels.csv")
        incident_series = load_incident_labels_from_s3(BUCKET, labels_key)

        # Align metrics and labels by timestamp
        df_metrics = df_metrics.set_index("timestamp")
        df_combined = df_metrics.join(incident_series, how="inner")
        df_combined = df_combined.reset_index()
        df_combined.columns = list(df_combined.columns[:-1]) + ["incident"]

        # 3. Create windows
        feature_cols = [c for c in df_combined.columns if c not in ("timestamp", "incident")]
        X_raw, y = create_sliding_window_dataset(
            df_combined, W=W, H=H, feature_cols=feature_cols
        )

        # Chronological split 80/20
        split = int(len(y) * 0.80)
        X_train, y_train = engineer_features(X_raw[:split]), y[:split]
        X_val,   y_val   = engineer_features(X_raw[split:]), y[split:]

        # 4. Train
        model = train(X_train, y_train, n_estimators=300, max_depth=5)

        # 5. Select threshold
        y_prob_val = model.predict_proba(X_val)[:, 1]
        threshold, metrics = select_threshold(y_val, y_prob_val, TARGET_RECALL)

        logger.info(f"Threshold={threshold:.4f} | recall={metrics['recall']:.4f} | "
                    f"fpr={metrics['fpr']:.4f}")

        # 6. Persist
        save_model_to_s3(model, threshold, metrics, version)
        publish_training_metrics(metrics, version)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "version": version,
                "threshold": threshold,
                "metrics": metrics,
            }),
        }

    except Exception as e:
        logger.exception("Retraining failed")
        return {"statusCode": 500, "body": str(e)}
