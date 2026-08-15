"""
tests/test_api.py
====================
Integration tests for the FastAPI serving layer, using FastAPI's
TestClient (starlette) — exercises the real app, including the lifespan
bootstrap, with no network calls or mocking.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402


def test_health_and_metrics():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["model_loaded"] is True

        r = client.get("/model/metrics")
        assert r.status_code == 200
        assert "roc_auc" in r.json()


def test_score_single_transaction():
    with TestClient(app) as client:
        r = client.post("/score", json={
            "Transaction_ID": "TXN000001", "User_ID": 1, "Source_Currency": "USD",
            "Target_Currency": "GBP", "Amount_USD": 200.0, "Device_IP": "24.1.1.1",
        })
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert body["action"] in ("Approved", "Flagged for Compliance Review", "Blocked")
        assert body["inference_latency_ms"] > 0


def test_score_rejects_invalid_amount():
    with TestClient(app) as client:
        r = client.post("/score", json={
            "Transaction_ID": "BAD", "User_ID": 1, "Source_Currency": "USD",
            "Target_Currency": "GBP", "Amount_USD": -10.0, "Device_IP": "1.1.1.1",
        })
        assert r.status_code == 422  # Pydantic gt=0 validation


def test_score_batch():
    with TestClient(app) as client:
        payload = {"transactions": [
            {"Transaction_ID": f"TXN{i:04d}", "User_ID": i, "Source_Currency": "USD",
             "Target_Currency": "EUR", "Amount_USD": 50.0 + i, "Device_IP": "24.2.2.2"}
            for i in range(5)
        ]}
        r = client.post("/score/batch", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["batch_size"] == 5
        assert len(body["results"]) == 5


def test_score_batch_rejects_empty_list():
    with TestClient(app) as client:
        r = client.post("/score/batch", json={"transactions": []})
        assert r.status_code == 400


def test_liquidity_state_and_tick():
    with TestClient(app) as client:
        r = client.get("/liquidity/state")
        assert r.status_code == 200
        pools = r.json()
        assert len(pools) == 8
        currencies = {p["currency"] for p in pools}
        assert "USD" in currencies and "INR" in currencies

        r = client.post("/liquidity/tick", json={"outflows": {"INR": 10000.0}})
        assert r.status_code == 200
        inr = [p for p in r.json() if p["currency"] == "INR"][0]
        assert "reorder_point" in inr and "fee_multiplier" in inr
