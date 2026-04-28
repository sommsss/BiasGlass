import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from fairlearn.postprocessing import ThresholdOptimizer


def train_model(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if y.dtype == "object":
        y = y.astype("category").cat.codes

    unique_vals = sorted(y.unique())
    if len(unique_vals) != 2:
        raise ValueError("Target column must be binary (2 unique values only)")

    y = (y == unique_vals[-1]).astype(int)

    X = pd.get_dummies(X, drop_first=True)
    X = X.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_test, y_pred, X_test, model, X_train, y_train


def train_mitigated_model(X_train, y_train, sensitive_train, X_test, sensitive_test):
    base_model = RandomForestClassifier(n_estimators=50, random_state=42)
    base_model.fit(X_train, y_train)

    mitigator = ThresholdOptimizer(
        estimator=base_model,
        constraints="demographic_parity",
        predict_method="predict_proba",
        objective="balanced_accuracy_score"
    )
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
    y_pred_mitigated = mitigator.predict(X_test, sensitive_features=sensitive_test)
    return y_pred_mitigated


def get_shap_values(model, X_train, X_test, max_samples=200):
    try:
        import shap

        # Force float — SHAP cannot handle object dtype
        X_train_f = X_train.astype(float)
        X_test_f = X_test.astype(float)

        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="interventional",
            data=X_train_f.sample(min(100, len(X_train_f)), random_state=42)
        )
        sample = X_test_f.sample(min(max_samples, len(X_test_f)), random_state=42)
        shap_values = explainer.shap_values(sample, check_additivity=False)

        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_vals = shap_values[:, :, 1]
        else:
            shap_vals = shap_values

        mean_abs_shap = pd.Series(
            np.abs(shap_vals).mean(axis=0),
            index=sample.columns
        ).sort_values(ascending=False)

        return mean_abs_shap, sample, shap_vals
    except ImportError:
        return None, None, None