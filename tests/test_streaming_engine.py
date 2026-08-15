"""
tests/test_streaming_engine.py
=================================
Tests the incremental (online) feature computation used by the live
dashboard and the FastAPI serving layer. The key property: features
computed incrementally, one transaction at a time, must reflect only
prior state — the same no-leakage guarantee tested for the batch path in
test_feature_engineering.py, but exercised through the streaming API.
"""
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from streaming_engine import LiveTransactionSimulator, StreamingFeatureStore  # noqa: E402


def test_streaming_velocity_matches_expected_prior_count():
    store = StreamingFeatureStore()
    base = datetime(2026, 1, 1, 12, 0, 0)

    r1 = store.process({
        "User_ID": 1, "Timestamp": base, "Amount_USD": 100.0,
        "Device_IP": "24.1.1.1", "Source_Currency": "USD", "Target_Currency": "GBP",
    })
    r2 = store.process({
        "User_ID": 1, "Timestamp": base + timedelta(minutes=10), "Amount_USD": 120.0,
        "Device_IP": "24.1.1.1", "Source_Currency": "USD", "Target_Currency": "GBP",
    })
    r3 = store.process({
        "User_ID": 1, "Timestamp": base + timedelta(minutes=20), "Amount_USD": 9500.0,
        "Device_IP": "24.1.1.1", "Source_Currency": "USD", "Target_Currency": "GBP",
    })

    assert r1["txn_count_1h"] == 0   # first-ever transaction for this user
    assert r2["txn_count_1h"] == 1   # sees only r1
    assert r3["txn_count_1h"] == 2   # sees r1 and r2, never itself


def test_device_fanout_detects_ring_pattern():
    store = StreamingFeatureStore()
    base = datetime(2026, 1, 1, 12, 0, 0)
    ring_ip = "185.1.1.1"

    results = []
    for i, uid in enumerate([10, 11, 12, 13]):
        r = store.process({
            "User_ID": uid, "Timestamp": base + timedelta(minutes=i * 2),
            "Amount_USD": 500.0, "Device_IP": ring_ip,
            "Source_Currency": "USD", "Target_Currency": "GBP",
        })
        results.append(r)

    # the 4th transaction on the shared device should see 3 distinct prior users
    assert results[3]["distinct_users_per_ip_1h"] == 3
    # the very first transaction on that device should see 0 prior users
    assert results[0]["distinct_users_per_ip_1h"] == 0


def test_geo_mismatch_flag():
    store = StreamingFeatureStore()
    now = datetime.now()
    matching = store.process({
        "User_ID": 1, "Timestamp": now, "Amount_USD": 50.0,
        "Device_IP": "24.9.9.9", "Source_Currency": "USD", "Target_Currency": "GBP",
    })
    mismatched = store.process({
        "User_ID": 2, "Timestamp": now, "Amount_USD": 50.0,
        "Device_IP": "103.9.9.9", "Source_Currency": "USD", "Target_Currency": "GBP",
    })
    assert matching["geo_mismatch"] == 0
    assert mismatched["geo_mismatch"] == 1


def test_live_simulator_produces_valid_transactions():
    sim = LiveTransactionSimulator(n_users=50, seed=42, fraud_probability=0.1)
    batch = sim.next_batch(100)
    assert len(batch) == 100
    for txn in batch:
        assert txn["Amount_USD"] > 0
        assert txn["Source_Currency"] != txn["Target_Currency"]
        assert txn["is_fraud"] in (0, 1)

    fraud_rate = sum(t["is_fraud"] for t in batch) / len(batch)
    # with fraud_probability=0.1 and burst patterns, expect a meaningfully
    # elevated (but not runaway) fraud rate over 100 transactions
    assert 0.0 <= fraud_rate <= 0.6
