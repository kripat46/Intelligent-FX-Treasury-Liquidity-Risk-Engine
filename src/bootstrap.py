"""
bootstrap.py
=============
Builds every artifact the dashboard needs, from scratch, at a scale tuned
to finish in a few seconds (rather than the multi-minute full-scale run
used for the standalone analysis scripts). This is what lets the
dashboard be genuinely self-contained on a fresh Streamlit Community
Cloud container: there is no pre-committed dataset in the repo — the app
generates and trains its own on first load, deterministically (fixed
seed), and caches the result for the life of the container.

Run standalone:
    python src/bootstrap.py

Or imported by dashboard/app.py inside an `st.cache_resource` function so
it only runs once per running app instance, not once per user session.
"""
from __future__ import annotations

import json
import os

import joblib

from data_generator import GeneratorConfig, TransactionStreamGenerator
from feature_engineering import build_features
from fraud_model import FraudDetectionEngine
from liquidity_optimizer import LiquidityOptimizer, estimate_pool_parameters_from_transactions

DEMO_CONFIG = GeneratorConfig(
    n_users=1200,
    n_days=10,
    base_daily_txn_rate=800,
    fraud_rate=0.02,
    random_seed=42,
)

STOCKOUT_CURRENCIES_TO_SHRINK = ("INR", "NGN")  # start these pools under-stocked for a live demo


def run_bootstrap(data_dir: str = "data") -> dict:
    os.makedirs(data_dir, exist_ok=True)

    # 1. synthetic transaction stream ------------------------------------
    gen = TransactionStreamGenerator(DEMO_CONFIG)
    raw = gen.generate()
    raw.to_csv(os.path.join(data_dir, "transactions.csv"), index=False)

    # 2. feature engineering ----------------------------------------------
    featured = build_features(raw)

    # 3. fraud model --------------------------------------------------------
    engine = FraudDetectionEngine()
    metrics = engine.fit(featured)
    with open(os.path.join(data_dir, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    scored = engine.score_batch(featured)
    keep_cols = [
        "Transaction_ID", "Timestamp", "User_ID", "Source_Currency",
        "Target_Currency", "Amount_USD", "Velocity_1H", "amount_zscore",
        "geo_mismatch", "is_fraud", "fraud_pattern",
        "fraud_probability", "action",
    ]
    scored[keep_cols].to_csv(os.path.join(data_dir, "scored_transactions.csv"), index=False)

    fi = engine.feature_importance()
    fi.to_csv(os.path.join(data_dir, "feature_importance.csv"), index=False)

    joblib.dump(engine, os.path.join(data_dir, "fraud_engine.joblib"))

    corridor = featured["Source_Currency"] + "->" + featured["Target_Currency"]
    corridor_freq = corridor.value_counts(normalize=True).to_dict()
    with open(os.path.join(data_dir, "corridor_frequency.json"), "w") as f:
        json.dump(corridor_freq, f, indent=2)

    # 4. liquidity optimization ------------------------------------------
    pools = [
        estimate_pool_parameters_from_transactions(raw, ccy)
        for ccy in ["USD", "INR", "GBP", "EUR", "PHP", "NGN", "BRL", "AUD"]
    ]
    for p in pools:
        if p.currency in STOCKOUT_CURRENCIES_TO_SHRINK:
            p.current_level *= 0.35

    optimizer = LiquidityOptimizer(pools)
    liq_df = optimizer.to_dataframe()
    liq_df.to_json(os.path.join(data_dir, "liquidity_state.json"), orient="records", indent=2)

    # also persist the raw PoolParameters (not just the derived results) so
    # the dashboard's live simulation can keep evolving current_level and
    # re-derive reorder_point / safety_stock / fee_multiplier on the fly
    pool_params = [p.__dict__ for p in pools]
    with open(os.path.join(data_dir, "pool_parameters.json"), "w") as f:
        json.dump(pool_params, f, indent=2)

    return {
        "n_transactions": len(raw),
        "metrics": metrics,
        "n_pools": len(pools),
    }


if __name__ == "__main__":
    result = run_bootstrap()
    print(json.dumps(result, indent=2, default=str))
