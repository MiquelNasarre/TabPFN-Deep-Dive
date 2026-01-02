"""
Smoke test for TabPFN environment.

What this script tests:
1) Core library imports
2) End-to-end TabPFN run:
   - model initialization
   - weight download / loading
   - fit on a small dataset
   - prediction
2) Basic timing (fit + predict latency on CPU)

This is NOT a benchmark, just a sanity check.
"""

import time
import sys
import platform

import numpy as np
import torch
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier


def run_local_smoke_test():

    print("=" * 60)
    print("TABPFN ENVIRONMENT SMOKE TEST")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1) Print environment info
    # ------------------------------------------------------------------
    print("\n[Environment]")
    print(f"Python version : {sys.version.split()[0]}")
    print(f"Platform       : {platform.platform()}")
    print(f"NumPy          : {np.__version__}")
    print(f"scikit-learn   : {sklearn.__version__}")
    print(f"PyTorch        : {torch.__version__}")
    print(f"TabPFN         : imported successfully")

    # ------------------------------------------------------------------
    # 2) Load example dataset used in TabPFN website
    # ------------------------------------------------------------------
    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    print("\n[Data]")
    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shape : X={X_test.shape}, y={y_test.shape}")

    # ------------------------------------------------------------------
    # 3) Initialize TabPFN (CPU mode)
    #    This will trigger weight loading on first use.
    # ------------------------------------------------------------------
    model = TabPFNClassifier(device="cpu")

    # ------------------------------------------------------------------
    # 4) Fit the model (timed)
    # ------------------------------------------------------------------
    print("\n[Model] Fitting TabPFN...")
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    # ------------------------------------------------------------------
    # 5) Predict (timed)
    # ------------------------------------------------------------------
    print("[Model] Predicting...")
    t0 = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - t0

    # ------------------------------------------------------------------
    # 6) Accuracy sanity check
    # ------------------------------------------------------------------
    accuracy = (y_pred == y_test).mean()

    print("\n[Results]")
    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Fit time (s)      : {fit_time:.2f}")
    print(f"Pred time (s)     : {pred_time:.2f}")
    print(f"Test samples      : {len(y_test)}")

    print("\nLocal smoke test completed successfully ✔")
    print("=" * 60)


if __name__ == "__main__":
    run_local_smoke_test()