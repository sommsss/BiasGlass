import pandas as pd
import numpy as np
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame,
    selection_rate,
    false_positive_rate,
    false_negative_rate,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score
 
 
def compute_metrics(y_true, y_pred, sensitive_features):
    dp = demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive_features
    )
    eo = equalized_odds_difference(
        y_true, y_pred, sensitive_features=sensitive_features
    )
 
    # Group-wise breakdown
    mf = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )
 
    group_rates = mf.by_group.to_dict()
    sel_rates = group_rates["selection_rate"]
 
    # Disparate Impact Ratio (80% rule — legal standard)
    if len(sel_rates) >= 2:
        max_rate = max(sel_rates.values())
        min_rate = min(sel_rates.values())
        disparate_impact_ratio = (min_rate / max_rate) if max_rate > 0 else 1.0
    else:
        disparate_impact_ratio = 1.0
 
    # Overall accuracy
    overall_accuracy = accuracy_score(y_true, y_pred)
 
    # Per-group accuracy
    mf_acc = MetricFrame(
        metrics={"accuracy": accuracy_score},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )
    group_accuracy = mf_acc.by_group.to_dict()["accuracy"]
 
    # Bias Risk Score (0–100, higher = more biased)
    dp_score = min(abs(dp) / 0.3, 1.0) * 40      # max 40 pts
    eo_score = min(abs(eo) / 0.3, 1.0) * 30      # max 30 pts
    di_score = (1 - min(disparate_impact_ratio, 1.0)) * 30  # max 30 pts (lower DI = more bias)
    bias_risk_score = round(dp_score + eo_score + di_score, 1)
 
    if bias_risk_score < 25:
        bias_risk_label = "Low"
        bias_risk_color = "green"
    elif bias_risk_score < 55:
        bias_risk_label = "Moderate"
        bias_risk_color = "orange"
    else:
        bias_risk_label = "High"
        bias_risk_color = "red"
 
    return {
        "demographic_parity_difference": round(dp, 4),
        "equalized_odds_difference": round(eo, 4),
        "disparate_impact_ratio": round(disparate_impact_ratio, 4),
        "overall_accuracy": round(overall_accuracy, 4),
        "group_selection_rates": group_rates,
        "group_accuracy": group_accuracy,
        "bias_risk_score": bias_risk_score,
        "bias_risk_label": bias_risk_label,
        "bias_risk_color": bias_risk_color,
    }
 
 
def compute_intersectional_metrics(y_true, y_pred, sensitive_features_df):
    """
    Takes a DataFrame of multiple sensitive columns, creates intersectional groups,
    and computes selection rate per intersection.
    """
    intersectional = sensitive_features_df.apply(
        lambda row: " & ".join(row.astype(str)), axis=1
    )
    mf = MetricFrame(
        metrics={"selection_rate": selection_rate},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=intersectional,
    )
    return mf.by_group.to_dict()["selection_rate"]
 