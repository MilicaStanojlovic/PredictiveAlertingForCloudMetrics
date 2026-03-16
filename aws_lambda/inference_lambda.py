"""
AWS Lambda: Real-time inference and alert triggering (runs every minute).

Architecture:
  - Triggered by EventBridge Scheduler (rate: 1 minute)
  - Fetches last W minutes of CloudWatch metrics
  - Runs model inference
  - If predicted risk >= threshold → publishes alert to SNS / CloudWatch alarm
  - Implements alert suppression to avoid repeated alerts for same incident

Environment variables:
  METRICS_BUCKET    S3 bucket where model artifacts are stored
  MODEL_KEY_PREFIX  S3 key prefix for model artifacts
  SNS_ALERT_ARN     SNS topic ARN for alert notifications
  CLOUDWATCH_NS     CloudWatch namespace for publishing risk scores
  WINDOW_W          Lookback window size (must match training)
  SERVICE_NAMESPACE AWS namespace for CloudWatch metrics to fetch
  METRIC_NAMES      Comma-separated metric names
  DIMENSIONS        JSON-encoded CloudWatch dimensions
  SUPPRESSION_TTL   Seconds to suppress duplicate alerts (default: 600 = 10 min)
"""

import json
import os
import io
import logging
from datetime import datetime, timedelta, timezone

import boto3
import numpy as np
import joblib

logger = logging.getLogger()
logger.setLevel(logging.INFO)

from src.features import engineer_features

S3   = boto3.client("s3")
SNS  = boto3.client("sns")
CW   = boto3.client("cloudwatch")
SSM  = boto3.client("ssm")  # for alert suppression state

BUCKET        = os.environ["METRICS_BUCKET"]
MODEL_PREFIX  = os.environ.get("MODEL_KEY_PREFIX", "incident-predictor/models/")
SNS_ARN       = os.environ["SNS_ALERT_ARN"]
CW_NAMESPACE  = os.environ.get("CLOUDWATCH_NS", "IncidentPredictor")
W             = int(os.environ.get("WINDOW_W", "30"))
SUPPRESSION_TTL = int(os.environ.get("SUPPRESSION_TTL", "600"))

# Module-level model cache (persists across warm Lambda invocations)
_MODEL_CACHE = {}


def load_model():
    """Load model from S3, with in-memory cache for warm starts."""
    key = f"{MODEL_PREFIX}latest/model.joblib"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    obj = S3.get_object(Bucket=BUCKET, Key=key)
    artifact = joblib.load(io.BytesIO(obj["Body"].read()))
    _MODEL_CACHE[key] = artifact
    logger.info(f"Model loaded from s3://{BUCKET}/{key}, version={artifact.get('version')}")
    return artifact


def fetch_recent_metrics(metric_names: list, namespace: str, dimensions: list) -> np.ndarray:
    """
    Fetch the last W minutes of CloudWatch metrics.
    Returns array of shape (W, n_metrics) or None if data is insufficient.
    """
    end_time   = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=W + 5)  # extra buffer

    queries = []
    for i, name in enumerate(metric_names):
        queries.append({
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
        })

    response = CW.get_metric_data(
        MetricDataQueries=queries,
        StartTime=start_time,
        EndTime=end_time,
        ScanBy="TimestampAscending",
    )

    # Parse and align into (W, n_metrics) matrix
    series = {}
    for result in response["MetricDataResults"]:
        idx = int(result["Id"][1:])
        # Take last W values
        values = result["Values"][-W:]
        if len(values) < W:
            values = [values[0]] * (W - len(values)) + list(values)  # pad with first value
        series[idx] = np.array(values, dtype=np.float32)

    if not series:
        return None

    n = len(metric_names)
    matrix = np.stack([series.get(i, np.zeros(W, dtype=np.float32)) for i in range(n)], axis=1)
    return matrix  # shape (W, n_metrics)


def is_suppressed(ssm_param: str) -> bool:
    """Check if alerts are currently suppressed (debounce)."""
    try:
        resp = SSM.get_parameter(Name=ssm_param)
        suppressed_until = float(resp["Parameter"]["Value"])
        return datetime.now(timezone.utc).timestamp() < suppressed_until
    except SSM.exceptions.ParameterNotFound:
        return False


def set_suppression(ssm_param: str, ttl_seconds: int):
    """Set alert suppression for ttl_seconds."""
    until = datetime.now(timezone.utc).timestamp() + ttl_seconds
    SSM.put_parameter(Name=ssm_param, Value=str(until), Type="String", Overwrite=True)


def publish_alert(risk_score: float, threshold: float, metric_values: np.ndarray):
    """Send alert to SNS topic."""
    message = {
        "alert_type": "IncidentPrediction",
        "risk_score": round(float(risk_score), 4),
        "threshold": round(float(threshold), 4),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": (
            f"Predicted incident risk {risk_score:.1%} exceeds threshold {threshold:.1%}. "
            f"Investigate metrics immediately."
        ),
    }
    SNS.publish(
        TopicArn=SNS_ARN,
        Subject="⚠️ Incident Predicted",
        Message=json.dumps(message, indent=2),
    )
    logger.warning(f"ALERT published: risk={risk_score:.4f} threshold={threshold:.4f}")


def publish_risk_metric(risk_score: float):
    """Publish predicted risk score to CloudWatch for dashboarding."""
    CW.put_metric_data(
        Namespace=CW_NAMESPACE,
        MetricData=[{
            "MetricName": "PredictedIncidentRisk",
            "Value": risk_score,
            "Unit": "None",
            "Timestamp": datetime.now(timezone.utc),
        }],
    )


def lambda_handler(event, context):
    """Entry point for the inference Lambda."""
    logger.info("Inference Lambda triggered")

    try:
        # 1. Load model
        artifact = load_model()
        model     = artifact["model"]
        threshold = artifact["threshold"]

        # 2. Fetch metrics
        metric_names = os.environ.get(
            "METRIC_NAMES", "CPUUtilization,Latency,ErrorRate,RequestCount"
        ).split(",")
        namespace  = os.environ.get("SERVICE_NAMESPACE", "AWS/ApplicationELB")
        dimensions = json.loads(os.environ.get("DIMENSIONS", "[]"))

        matrix = fetch_recent_metrics(metric_names, namespace, dimensions)
        if matrix is None or matrix.shape[0] < W:
            logger.warning("Insufficient metric data for inference")
            return {"statusCode": 200, "body": "insufficient_data"}

        # 3. Engineer features and predict
        X_raw = matrix[np.newaxis, :, :]  # (1, W, n_metrics)
        X     = engineer_features(X_raw, feature_names=metric_names)
        risk_score = float(model.predict_proba(X)[0, 1])

        logger.info(f"Predicted risk score: {risk_score:.4f} (threshold={threshold:.4f})")

        # 4. Publish risk score to CloudWatch
        publish_risk_metric(risk_score)

        # 5. Trigger alert if risk exceeds threshold (with suppression)
        suppression_param = "/incident-predictor/alert-suppressed"
        if risk_score >= threshold:
            if not is_suppressed(suppression_param):
                publish_alert(risk_score, threshold, matrix)
                set_suppression(suppression_param, SUPPRESSION_TTL)
            else:
                logger.info("Alert suppressed (debounce active)")

        return {
            "statusCode": 200,
            "body": json.dumps({
                "risk_score": risk_score,
                "threshold": threshold,
                "alert_triggered": risk_score >= threshold,
            }),
        }

    except Exception as e:
        logger.exception("Inference failed")
        return {"statusCode": 500, "body": str(e)}
