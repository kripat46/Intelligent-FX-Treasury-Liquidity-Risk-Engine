"""
tests/test_fraud_model.py
============================
Tests the threshold-policy layer in isolation from the trained model —
this is the part of the fraud engine with the clearest, checkable
business logic (ordering, monotonicity, recall behavior).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fraud_model import FraudDetectionEngine, ThresholdPolicy  # noqa: E402


def test_threshold_policy_ordering_enforced():
    policy = ThresholdPolicy(t_flag=0.5, t_block=0.9)
    proba = np.array([0.1, 0.5, 0.7, 0.95])
    actions = policy.classify(proba)
    assert list(actions) == [
        "Approved", "Flagged for Compliance Review",
        "Flagged for Compliance Review", "Blocked",
    ]


def test_threshold_policy_is_monotonic():
    """Higher probability must never result in a less severe action."""
    policy = ThresholdPolicy(t_flag=0.3, t_block=0.8)
    severity = {"Approved": 0, "Flagged for Compliance Review": 1, "Blocked": 2}
    proba = np.linspace(0, 1, 50)
    actions = policy.classify(proba)
    severities = [severity[a] for a in actions]
    assert severities == sorted(severities), "action severity must be non-decreasing in probability"


def test_tune_thresholds_respects_recall_floor():
    """Synthetic scores where fraud cleanly separates from non-fraud at 0.6:
    the tuned t_flag should sit at or below that separation point so the
    recall SLA (TARGET_RECALL) is achievable."""
    rng = np.random.default_rng(0)
    n = 2000
    y = np.zeros(n, dtype=int)
    y[:40] = 1  # 2% fraud rate
    proba = np.where(y == 1,
                      rng.uniform(0.6, 1.0, n),
                      rng.uniform(0.0, 0.5, n))
    policy = FraudDetectionEngine._tune_thresholds(y, proba)
    caught = (proba >= policy.t_flag).astype(int)
    recall = caught[y == 1].sum() / y.sum()
    assert recall >= 0.85, f"recall {recall} fell short of the SLA the policy is tuned for"
    assert policy.t_block >= policy.t_flag
