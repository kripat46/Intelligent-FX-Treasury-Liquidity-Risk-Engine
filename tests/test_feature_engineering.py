"""
tests/test_feature_engineering.py
====================================
Correctness tests for the batch feature pipeline. The single property that
matters most here is **no leakage**: a rolling feature must never be
influenced by the row it's being computed for.
"""
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from feature_engineering import FEATURE_COLUMNS, build_features  # noqa: E402


def _toy_transactions() -> pd.DataFrame:
    base = datetime(2026, 1, 1, 12, 0, 0)
    rows = [
        # user 1: three transactions 10 minutes apart -> the 3rd should see
        # txn_count_1h == 2 (the two prior), not 3.
        {"Transaction_ID": "T1", "Timestamp": base, "User_ID": 1,
         "Source_Currency": "USD", "Target_Currency": "GBP", "Amount_USD": 100.0,
         "Device_IP": "24.1.1.1", "Velocity_1H": 0, "is_fraud": 0, "fraud_pattern": "none"},
        {"Transaction_ID": "T2", "Timestamp": base + timedelta(minutes=10), "User_ID": 1,
         "Source_Currency": "USD", "Target_Currency": "GBP", "Amount_USD": 120.0,
         "Device_IP": "24.1.1.1", "Velocity_1H": 0, "is_fraud": 0, "fraud_pattern": "none"},
        {"Transaction_ID": "T3", "Timestamp": base + timedelta(minutes=20), "User_ID": 1,
         "Source_Currency": "USD", "Target_Currency": "GBP", "Amount_USD": 9500.0,
         "Device_IP": "24.1.1.1", "Velocity_1H": 0, "is_fraud": 1, "fraud_pattern": "structuring"},
        # user 2: a single, unrelated transaction
        {"Transaction_ID": "T4", "Timestamp": base + timedelta(minutes=5), "User_ID": 2,
         "Source_Currency": "INR", "Target_Currency": "USD", "Amount_USD": 300.0,
         "Device_IP": "103.5.5.5", "Velocity_1H": 0, "is_fraud": 0, "fraud_pattern": "none"},
    ]
    return pd.DataFrame(rows)


def test_no_self_leakage_in_rolling_count():
    df = build_features(_toy_transactions())
    t3 = df[df["Transaction_ID"] == "T3"].iloc[0]
    # T3 should see exactly 2 prior transactions from user 1 in the trailing hour,
    # NOT 3 (which would mean it counted itself).
    assert t3["txn_count_1h"] == 2
    assert t3["txn_sum_1h"] == pytest.approx(220.0)


def test_near_threshold_flag_fires_correctly():
    df = build_features(_toy_transactions())
    t3 = df[df["Transaction_ID"] == "T3"].iloc[0]
    assert t3["near_threshold_flag"] == 1
    t1 = df[df["Transaction_ID"] == "T1"].iloc[0]
    assert t1["near_threshold_flag"] == 0


def test_all_feature_columns_present_and_numeric():
    df = build_features(_toy_transactions())
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"missing feature column: {col}"
        assert pd.api.types.is_numeric_dtype(df[col]), f"non-numeric feature column: {col}"


def test_feature_engineering_is_order_invariant_on_input():
    """Shuffling input row order must not change the resulting features,
    since build_features sorts by Timestamp internally."""
    df = _toy_transactions()
    shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    out_a = build_features(df).set_index("Transaction_ID").sort_index()
    out_b = build_features(shuffled).set_index("Transaction_ID").sort_index()
    pd.testing.assert_series_equal(out_a["txn_count_1h"], out_b["txn_count_1h"])
