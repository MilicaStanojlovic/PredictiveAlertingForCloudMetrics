"""
Incident-level evaluation metrics.

Window-level metrics (precision/recall on individual windows) are necessary
but not sufficient for an alerting system. What matters operationally is:
  1. Was at least one alert raised BEFORE each incident started?  (incident recall)
  2. How early was the first alert?  (lead time)
  3. How many alert bursts are false positives?  (alert precision)

This module computes these incident-aware metrics on top of per-window predictions.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict


@dataclass
class AlertResult:
    incident_idx: int
    incident_start: int        # time index
    incident_end: int
    severity: float
    first_alert_at: int = -1   # -1 means no alert raised
    lead_time_steps: int = 0   # steps before incident_start
    detected: bool = False


def evaluate_incidents(
    y_prob: np.ndarray,
    incidents: list,          # List[IncidentInterval] from data_generator
    W: int,
    H: int,
    threshold: float,
    pre_incident_window: int = None,  # how far before incident to count as "early warning"
) -> Tuple[List[AlertResult], Dict]:
    """
    Map per-window predictions back to incident-level detection.

    A window at position i covers future steps [i+W, i+W+H).
    An alert is raised when y_prob[i] >= threshold.
    An incident is *detected* if there exists an alert at position i such that
    i+W <= incident.start (alert fires before incident starts).

    Lead time = incident.start - (i + W) steps (positive → early warning).
    """
    if pre_incident_window is None:
        pre_incident_window = H

    alerts_at = np.where(y_prob >= threshold)[0]  # window indices with alerts

    results = []
    for idx, inc in enumerate(incidents):
        result = AlertResult(
            incident_idx=idx,
            incident_start=inc.start,
            incident_end=inc.end,
            severity=inc.severity,
        )

        # Find alerts that fire before the incident starts
        # Window i fires for future [i+W, i+W+H), so alert fires BEFORE incident if:
        # i+W <= inc.start  →  i <= inc.start - W
        early_alerts = alerts_at[alerts_at <= inc.start - W]

        # Also require alert is within a reasonable look-ahead window
        # i.e. alert is close enough to the incident to be meaningful
        # We define "meaningful" as: i + W >= inc.start - pre_incident_window
        # → i >= inc.start - W - pre_incident_window
        meaningful_early = early_alerts[
            early_alerts >= inc.start - W - pre_incident_window
        ]

        if len(meaningful_early) > 0:
            first_alert_window = meaningful_early[0]
            lead_time = inc.start - (first_alert_window + W)
            result.first_alert_at = int(first_alert_window)
            result.lead_time_steps = int(lead_time)
            result.detected = True

        results.append(result)

    # Aggregate
    detected = [r for r in results if r.detected]
    missed = [r for r in results if not r.detected]

    incident_recall = len(detected) / len(results) if results else 0.0
    lead_times = [r.lead_time_steps for r in detected]
    mean_lead = np.mean(lead_times) if lead_times else 0.0
    median_lead = np.median(lead_times) if lead_times else 0.0

    # False positive alert bursts: alert windows that don't precede any incident
    # Simple approximation: alerts not within [inc.start - W - H, inc.start - W]
    all_incident_alert_windows = set()
    for inc in incidents:
        for w in range(max(0, inc.start - W - pre_incident_window), inc.start - W + 1):
            all_incident_alert_windows.add(w)
    # Also mark alerts during incidents as TP (window overlaps incident)
    for inc in incidents:
        for w in range(max(0, inc.start - W), inc.end):
            all_incident_alert_windows.add(w)

    fp_alerts = len([a for a in alerts_at if a not in all_incident_alert_windows])
    total_alerts = len(alerts_at)

    summary = {
        "n_incidents": len(results),
        "n_detected": len(detected),
        "n_missed": len(missed),
        "incident_recall": round(incident_recall, 4),
        "mean_lead_time_steps": round(mean_lead, 2),
        "median_lead_time_steps": round(median_lead, 2),
        "total_alerts_raised": total_alerts,
        "fp_alert_windows": fp_alerts,
        "alert_noise_ratio": round(fp_alerts / (total_alerts + 1e-8), 4),
    }

    return results, summary


def print_incident_report(results: List[AlertResult], summary: Dict, sampling_interval_sec: int = 60):
    print("\n=== Incident-Level Evaluation ===")
    print(f"Total incidents:     {summary['n_incidents']}")
    print(f"Detected:            {summary['n_detected']} ({summary['incident_recall']:.1%})")
    print(f"Missed:              {summary['n_missed']}")
    print(f"Mean lead time:      {summary['mean_lead_time_steps'] * sampling_interval_sec / 60:.1f} min")
    print(f"Median lead time:    {summary['median_lead_time_steps'] * sampling_interval_sec / 60:.1f} min")
    print(f"Total alerts raised: {summary['total_alerts_raised']}")
    print(f"FP alert windows:    {summary['fp_alert_windows']}")
    print(f"Alert noise ratio:   {summary['alert_noise_ratio']:.2%}")
    print()
    for r in results:
        status = "✓ DETECTED" if r.detected else "✗ MISSED  "
        lead_min = r.lead_time_steps * sampling_interval_sec / 60
        lead_str = f"lead={lead_min:.0f}min" if r.detected else "no alert"
        print(f"  [{status}] incident #{r.incident_idx:03d} "
              f"t={r.incident_start}-{r.incident_end} "
              f"sev={r.severity:.2f} | {lead_str}")
