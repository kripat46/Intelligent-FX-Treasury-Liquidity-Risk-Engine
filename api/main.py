"""
api/main.py
=============
Production-style serving layer, decoupled from the Streamlit dashboard.

Why this exists as a separate service (not just more Streamlit code):
a dashboard is a human-facing view; a payments platform's actual risk
decision has to be callable synchronously from the transfer-creation code
path, with a stable contract, independent of whether anyone has a browser
tab open. This FastAPI app is that callable contract — the same
`FraudDetectionEngine` and `LivePoolBook` the dashboard visualizes are
served here as versioned HTTP endpoints, which is the shape a real
architecture takes: one shared model/artifact layer, multiple consumers
(a dashboard, a batch job, this API) reading from it independently.

Endpoints
---------
GET  /health                    liveness/readiness + artifact status
GET  /model/metrics             offline holdout metrics for the loaded model
POST /score                     score a single transaction (stateful: updates
                                 the process-local streaming feature store)
POST /score/batch               score a list of transactions in one call
GET  /liquidity/state           current (s, S) state for all currency pools
POST /liquidity/tick            advance the liquidity simulation by one tick

Run locally:
    uvicorn api.main:app --reload --port 8000
    open http://localhost:8000/docs   (interactive OpenAPI docs)
"""
from __future__ import annotations

import json
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))
sys.path.insert(0, SRC)

from feature_engineering import FEATURE_COLUMNS  # noqa: E402
from liquidity_optimizer import LivePoolBook, PoolParameters  # noqa: E402
from streaming_engine import StreamingFeatureStore  # noqa: E402


# ------------------------------------------------------------------ #
# Process-local state. In a real multi-instance deployment this
# in-memory feature store would be backed by a shared store (Redis /
# a feature-store service) so all replicas see the same rolling
# history — noted explicitly here rather than glossed over, since
# it's the honest limitation of running this as-is behind a
# horizontally-scaled Kubernetes Deployment (see k8s/README.md).
# ------------------------------------------------------------------ #
class AppState:
    engine = None
    feature_store: Optional[StreamingFeatureStore] = None
    pool_book: Optional[LivePoolBook] = None
    model_metrics: dict = {}


state = AppState()


def _ensure_artifacts():
    required = [
        "transactions.csv", "model_metrics.json", "fraud_engine.joblib",
        "corridor_frequency.json", "pool_parameters.json",
    ]
    if not all(os.path.exists(os.path.join(DATA_DIR, r)) for r in required):
        from bootstrap import run_bootstrap
        run_bootstrap(DATA_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_artifacts()
    state.engine = joblib.load(os.path.join(DATA_DIR, "fraud_engine.joblib"))
    corridor_freq = json.load(open(os.path.join(DATA_DIR, "corridor_frequency.json")))
    state.feature_store = StreamingFeatureStore(corridor_frequency=corridor_freq)
    pool_params = [PoolParameters(**d) for d in json.load(open(os.path.join(DATA_DIR, "pool_parameters.json")))]
    state.pool_book = LivePoolBook(pool_params, replenishment_lead_ticks=6)
    state.model_metrics = json.load(open(os.path.join(DATA_DIR, "model_metrics.json")))
    yield
    # no explicit teardown needed — process-local state is garbage collected


app = FastAPI(
    title="Treasury Liquidity & Fraud Risk API",
    description="Serving layer for the fraud scoring and (s, S) liquidity optimization engines.",
    version="1.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------ #
# Schemas
# ------------------------------------------------------------------ #
class TransactionIn(BaseModel):
    Transaction_ID: str = Field(..., examples=["TXN000001"])
    User_ID: int = Field(..., examples=[1042])
    Source_Currency: str = Field(..., examples=["USD"])
    Target_Currency: str = Field(..., examples=["INR"])
    Amount_USD: float = Field(..., gt=0, examples=[482.50])
    Device_IP: str = Field(..., examples=["24.11.201.4"])
    Timestamp: Optional[datetime] = None


class ScoreOut(BaseModel):
    Transaction_ID: str
    fraud_probability: float
    action: str
    inference_latency_ms: float


class BatchScoreRequest(BaseModel):
    transactions: list[TransactionIn]


class BatchScoreResponse(BaseModel):
    results: list[ScoreOut]
    batch_size: int
    avg_latency_ms: float


class TickRequest(BaseModel):
    outflows: dict[str, float] = Field(
        default_factory=dict,
        description="Currency -> outflow amount to apply this tick, e.g. {'INR': 50000}",
    )


# ------------------------------------------------------------------ #
# Core scoring helper (shared by /score and /score/batch)
# ------------------------------------------------------------------ #
def _score_transactions(txns: list[TransactionIn]) -> tuple[pd.DataFrame, float]:
    if state.engine is None or state.feature_store is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    raw_dicts = []
    for t in txns:
        d = t.model_dump()
        d["Timestamp"] = d["Timestamp"] or datetime.now()
        raw_dicts.append(d)

    featured_rows = [state.feature_store.process(d) for d in raw_dicts]
    batch_df = pd.DataFrame(featured_rows)

    X = batch_df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
    t0 = time.perf_counter()
    proba = state.engine.model.predict_proba(X)[:, 1]
    elapsed_ms = (time.perf_counter() - t0) * 1000
    actions = state.engine.policy.classify(proba)

    batch_df["fraud_probability"] = proba
    batch_df["action"] = actions
    return batch_df, elapsed_ms


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #
@app.get("/health")
def health():
    return {
        "status": "ok" if state.engine is not None else "loading",
        "model_loaded": state.engine is not None,
        "data_dir": DATA_DIR,
    }


@app.get("/model/metrics")
def model_metrics():
    if not state.model_metrics:
        raise HTTPException(status_code=503, detail="Metrics not loaded yet")
    return state.model_metrics


@app.post("/score", response_model=ScoreOut)
def score_transaction(txn: TransactionIn):
    """Scores a single transaction against the live streaming feature
    store (this call updates that user/device's rolling history, exactly
    as a real inbound-payment risk check would)."""
    batch_df, elapsed_ms = _score_transactions([txn])
    row = batch_df.iloc[0]
    return ScoreOut(
        Transaction_ID=row["Transaction_ID"],
        fraud_probability=round(float(row["fraud_probability"]), 6),
        action=row["action"],
        inference_latency_ms=round(elapsed_ms, 3),
    )


@app.post("/score/batch", response_model=BatchScoreResponse)
def score_batch(request: BatchScoreRequest):
    if not request.transactions:
        raise HTTPException(status_code=400, detail="transactions list is empty")
    batch_df, elapsed_ms = _score_transactions(request.transactions)
    results = [
        ScoreOut(
            Transaction_ID=row["Transaction_ID"],
            fraud_probability=round(float(row["fraud_probability"]), 6),
            action=row["action"],
            inference_latency_ms=round(elapsed_ms / len(batch_df), 4),
        )
        for _, row in batch_df.iterrows()
    ]
    return BatchScoreResponse(
        results=results,
        batch_size=len(results),
        avg_latency_ms=round(elapsed_ms / len(batch_df), 4),
    )


@app.get("/liquidity/state")
def liquidity_state():
    """Returns the current (s, S) state for every currency pool without
    advancing the simulation (a zero-outflow tick)."""
    snapshot = state.pool_book.tick({})
    return json.loads(snapshot.to_json(orient="records"))


@app.post("/liquidity/tick")
def liquidity_tick(request: TickRequest):
    """Advances the liquidity simulation by one tick given a dict of
    currency -> outflow amount, mutating pool levels and re-solving the
    reorder point / safety stock / fee multiplier for each pool."""
    snapshot = state.pool_book.tick(request.outflows)
    return json.loads(snapshot.to_json(orient="records"))
