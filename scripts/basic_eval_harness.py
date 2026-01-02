"""
Minimal TabPFN evaluation:
- split
- fit
- predict
- metric
Returns a dict with metrics + timings.
"""

import time
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score


def run_tabpfn_once(
    X,
    y,
    *,
    backend: str = "local",          # "local" or "cloud"
    task: str = "classification",    # "classification" or "regression"
    seed: int = 42,
    test_size: float = 0.3,
) -> dict:

    # ------------------------------------------------------------------
    # 1) Split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y if task == "classification" else None,
    )

    # ------------------------------------------------------------------
    # 2) Load model
    # ------------------------------------------------------------------
    if backend == "local":
        if task == "classification":
            from tabpfn import TabPFNClassifier
            model = TabPFNClassifier()
        else:
            from tabpfn import TabPFNRegressor
            model = TabPFNRegressor()

    else:
        from dotenv import load_dotenv
        if not load_dotenv(): # Load access token from .env file
            print("Warning: .env file not found or could not be loaded. Make sure PRIORLABS_API_KEY is set in the environment.")

        if task == "classification":
            from tabpfn_client import TabPFNClassifier
            model = TabPFNClassifier()
        else:
            from tabpfn_client import TabPFNRegressor
            model = TabPFNRegressor()

    # ------------------------------------------------------------------
    # 3) Fit
    # ------------------------------------------------------------------
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    # ------------------------------------------------------------------
    # 4) Predict
    # ------------------------------------------------------------------
    t0 = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - t0

    # ------------------------------------------------------------------
    # 5) Metric
    # ------------------------------------------------------------------
    if task == "classification":
        metric = accuracy_score(y_test, y_pred)
        metric_name = "accuracy"
    else:
        metric = r2_score(y_test, y_pred)
        metric_name = "r2"

    # ------------------------------------------------------------------
    # 6) Return result
    # ------------------------------------------------------------------
    return {
        "backend": backend,
        "task": task,
        "seed": seed,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "metric_name": metric_name,
        "metric_value": float(metric),
        "fit_time_sec": float(fit_time),
        "predict_time_sec": float(predict_time),
    }