"""
train_with_mlflow.py
=======================
Training entrypoint that wraps `FraudDetectionEngine.fit()` with MLflow
experiment tracking: params, metrics, and the model artifact are logged
to a run, so training runs are comparable and reproducible rather than
being judged from whatever printed to stdout during the last run someone
happened to keep.

Deliberately kept separate from `fraud_model.py`: the core engine has no
MLflow dependency, so it can be imported by the API/dashboard without
pulling in the tracking stack. This mirrors a common real-world split —
a lightweight model/serving library plus a separate, heavier training
harness that owns experiment tracking.

Usage:
    python src/train_with_mlflow.py                       # local file-store tracking (./mlruns)
    MLFLOW_TRACKING_URI=http://localhost:5000 python src/train_with_mlflow.py   # remote/dockerized tracking server
"""
from __future__ import annotations

import json
import os

import mlflow.xgboost
import pandas as pd

import mlflow
from data_generator import GeneratorConfig, TransactionStreamGenerator
from feature_engineering import build_features
from fraud_model import TARGET_BLOCK_PRECISION, TARGET_RECALL, FraudDetectionEngine

EXPERIMENT_NAME = "fx-treasury-fraud-detection"
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(_SRC_DIR), "data")


def main(data_dir: str = _DEFAULT_DATA_DIR, use_existing_data: bool = True):
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="xgboost-fraud-classifier") as run:
        # --- data ------------------------------------------------------------
        raw_path = os.path.join(data_dir, "transactions.csv")
        if use_existing_data and os.path.exists(raw_path):
            raw = pd.read_csv(raw_path)
        else:
            gen = TransactionStreamGenerator(GeneratorConfig())
            raw = gen.generate()
            os.makedirs(data_dir, exist_ok=True)
            raw.to_csv(raw_path, index=False)

        featured = build_features(raw)
        mlflow.log_param("n_rows", len(raw))
        mlflow.log_param("fraud_rate_actual", float(raw["is_fraud"].mean()))

        # --- model hyperparameters (mirrors FraudDetectionEngine.fit defaults,
        # logged explicitly here so they're comparable run-over-run even if
        # the engine's defaults are later changed) -----------------------------
        hyperparams = {
            "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
            "reg_lambda": 1.0, "target_recall_sla": TARGET_RECALL,
            "target_block_precision": TARGET_BLOCK_PRECISION,
        }
        mlflow.log_params(hyperparams)

        # --- train -------------------------------------------------------------
        engine = FraudDetectionEngine()
        metrics = engine.fit(featured)

        # --- log metrics (flatten the nested confusion_matrix / action_distribution
        # dicts, since MLflow metrics must be scalar) ----------------------------
        scalar_metrics = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        mlflow.log_metrics(scalar_metrics)
        mlflow.log_dict(metrics["confusion_matrix"], "confusion_matrix.json")
        mlflow.log_dict(metrics["action_distribution"], "action_distribution.json")

        # --- log the model artifact ------------------------------------------
        mlflow.xgboost.log_model(
            engine.model, name="model",
            registered_model_name=None,  # set to a string to auto-register, e.g. "fx-fraud-classifier"
        )

        # --- log feature importances as a table --------------------------------
        fi = engine.feature_importance()
        fi_path = os.path.join(data_dir, "feature_importance.csv")
        fi.to_csv(fi_path, index=False)
        mlflow.log_artifact(fi_path)

        print(f"MLflow run ID: {run.info.run_id}")
        print(f"Tracking URI:  {mlflow.get_tracking_uri()}")
        print(json.dumps(scalar_metrics, indent=2))

        return run.info.run_id, metrics


if __name__ == "__main__":
    main()
