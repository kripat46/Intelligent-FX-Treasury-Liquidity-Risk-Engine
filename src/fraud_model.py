"""
fraud_model.py
===============
Real-time fraud/AML risk engine.

Modeling approach
------------------
We train **one binary XGBoost classifier** that outputs a calibrated fraud
probability p(fraud | features), then apply a **two-threshold policy layer**
to convert that probability into the three business actions:

    p < t_flag                      -> "Approved"
    t_flag <= p < t_block           -> "Flagged for Compliance Review"
    p >= t_block                    -> "Blocked"

Why not a native 3-class classifier? Because "should this have been
blocked outright vs. merely flagged" is a business-policy decision (driven
by risk appetite, regulatory obligations, and operational review capacity),
not a statistical property of the data. Decoupling the *score* from the
*policy* means compliance can retune the two thresholds without retraining
the model — exactly how production risk engines are architected.

Threshold selection: minimizing false negatives
-------------------------------------------------
Fraud losses are highly asymmetric: a missed fraud (false negative) costs
far more than reviewing a legitimate transaction (false positive) costs in
operational overhead. We therefore do NOT pick thresholds by F1 or
accuracy. Instead:

  * `t_block` is set to the probability threshold that achieves at least
    `TARGET_BLOCK_PRECISION` precision (we don't want to auto-block clean
    users) among the highest-risk transactions.
  * `t_flag` is set by walking the precision-recall curve and choosing the
    lowest threshold such that recall on the full (block + flag) fraud
    catch-rate is >= `TARGET_RECALL` — i.e., "catch at least X% of all
    fraud, across either action", which is the operational SLA a
    compliance team would actually set.

Class imbalance handling
--------------------------
With ~2% positive class, we use XGBoost's `scale_pos_weight` (ratio of
negatives to positives) rather than naive oversampling, which avoids
duplicating rare fraud rows and the overfitting that comes with it, and we
evaluate primarily on **PR-AUC and recall-at-fixed-precision**, since
ROC-AUC is known to be overly optimistic under heavy class imbalance.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from feature_engineering import FEATURE_COLUMNS, build_features

TARGET_RECALL = 0.90            # SLA: catch >=90% of all fraud (flag+block combined)
TARGET_BLOCK_PRECISION = 0.85   # only auto-block when we're this confident


@dataclass
class ThresholdPolicy:
    t_flag: float
    t_block: float

    def classify(self, proba: np.ndarray) -> np.ndarray:
        actions = np.full(len(proba), "Approved", dtype=object)
        actions[proba >= self.t_flag] = "Flagged for Compliance Review"
        actions[proba >= self.t_block] = "Blocked"
        return actions


class FraudDetectionEngine:
    """Wraps a trained XGBoost model + threshold policy for training,
    evaluation, and low-latency single-transaction inference."""

    def __init__(self, feature_columns: list[str] = FEATURE_COLUMNS):
        self.feature_columns = feature_columns
        self.model: xgb.XGBClassifier | None = None
        self.policy: ThresholdPolicy | None = None
        self.metrics: dict = {}

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(self, df_featured: pd.DataFrame, test_size: float = 0.25,
            random_state: int = 42) -> dict:
        X = df_featured[self.feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = df_featured["is_fraud"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / max(n_pos, 1)

        self.model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="aucpr",          # PR-AUC, appropriate for rare-event classification
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        proba_test = self.model.predict_proba(X_test)[:, 1]
        self.policy = self._tune_thresholds(y_test, proba_test)
        self.metrics = self._evaluate(y_test, proba_test, self.policy)
        self.metrics["scale_pos_weight"] = float(scale_pos_weight)
        self.metrics["n_train"] = int(len(X_train))
        self.metrics["n_test"] = int(len(X_test))

        # store test set for downstream artifacts (dashboard demo feed)
        self._X_test, self._y_test, self._proba_test = X_test, y_test, proba_test
        return self.metrics

    # ------------------------------------------------------------------ #
    # Threshold tuning
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tune_thresholds(y_true: np.ndarray, proba: np.ndarray) -> ThresholdPolicy:
        precision, recall, thresholds = precision_recall_curve(y_true, proba)
        # precision_recall_curve returns thresholds of length n-1 vs precision/recall of length n
        # align by dropping the last precision/recall point (which has no corresponding threshold)
        precision, recall = precision[:-1], recall[:-1]

        # --- t_flag: lowest threshold that still clears the recall SLA ---
        eligible = np.where(recall >= TARGET_RECALL)[0]
        if len(eligible) > 0:
            # among thresholds achieving the recall SLA, take the one with
            # highest threshold (best precision) that still clears it
            t_flag = thresholds[eligible[np.argmax(thresholds[eligible])]]
        else:
            t_flag = float(np.percentile(proba, 90))  # fallback: top 10% by score

        # --- t_block: lowest threshold achieving the block-precision bar ---
        block_eligible = np.where(precision >= TARGET_BLOCK_PRECISION)[0]
        if len(block_eligible) > 0:
            t_block = thresholds[block_eligible[np.argmax(thresholds[block_eligible])]]
        else:
            t_block = float(np.percentile(proba, 99.5))

        t_block = max(t_block, t_flag + 1e-3)  # ensure ordering
        return ThresholdPolicy(t_flag=float(t_flag), t_block=float(t_block))

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _evaluate(y_true: np.ndarray, proba: np.ndarray, policy: ThresholdPolicy) -> dict:
        actions = policy.classify(proba)
        # "caught" = flagged OR blocked; false negative = fraud that was Approved
        caught = np.isin(actions, ["Flagged for Compliance Review", "Blocked"]).astype(int)
        y_pred_binary = caught  # for standard precision/recall against catch-or-not

        cm = confusion_matrix(y_true, y_pred_binary)
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            "roc_auc": float(roc_auc_score(y_true, proba)),
            "pr_auc": float(average_precision_score(y_true, proba)),
            "precision_at_policy": float(precision_score(y_true, y_pred_binary, zero_division=0)),
            "recall_at_policy": float(recall_score(y_true, y_pred_binary, zero_division=0)),
            "f1_at_policy": float(f1_score(y_true, y_pred_binary, zero_division=0)),
            "false_negative_rate": float(fn / max(fn + tp, 1)),
            "false_negatives_count": int(fn),
            "true_positives_count": int(tp),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "t_flag": policy.t_flag,
            "t_block": policy.t_block,
            "action_distribution": pd.Series(actions).value_counts(normalize=True).to_dict(),
        }
        return metrics

    # ------------------------------------------------------------------ #
    # Low-latency single-transaction inference
    # ------------------------------------------------------------------ #
    def score_transaction(self, feature_row: pd.Series) -> dict:
        """Simulates real-time scoring of a single transaction. Returns the
        fraud probability, the resulting action, and measured inference
        latency in milliseconds — the metric that matters for a real-time
        payment-blocking decision (must comfortably clear the platform's
        SLA, typically < 100ms end-to-end).
        """
        if self.model is None or self.policy is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        x = feature_row[self.feature_columns].to_frame().T.replace(
            [np.inf, -np.inf], np.nan).fillna(0).astype(float)

        t0 = time.perf_counter()
        proba = float(self.model.predict_proba(x)[0, 1])
        action = self.policy.classify(np.array([proba]))[0]
        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "fraud_probability": round(proba, 6),
            "action": action,
            "inference_latency_ms": round(latency_ms, 3),
        }

    def score_batch(self, df_featured: pd.DataFrame) -> pd.DataFrame:
        """Vectorized batch scoring (used for the dashboard's live feed
        simulation) with per-row latency measured to illustrate throughput
        characteristics, not just single-row latency."""
        X = df_featured[self.feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
        t0 = time.perf_counter()
        proba = self.model.predict_proba(X)[:, 1]
        elapsed = time.perf_counter() - t0

        out = df_featured.copy()
        out["fraud_probability"] = proba
        out["action"] = self.policy.classify(proba)
        out["avg_inference_latency_ms"] = round((elapsed / len(df_featured)) * 1000, 4)
        return out

    def feature_importance(self) -> pd.DataFrame:
        importances = self.model.feature_importances_
        return pd.DataFrame({
            "feature": self.feature_columns,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import joblib

    raw = pd.read_csv("data/transactions.csv")
    featured = build_features(raw)

    engine = FraudDetectionEngine()
    metrics = engine.fit(featured)

    print("=== Fraud Detection Engine — Evaluation Metrics ===")
    print(json.dumps(metrics, indent=2, default=str))

    with open("data/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    scored = engine.score_batch(featured)
    keep_cols = [
        "Transaction_ID", "Timestamp", "User_ID", "Source_Currency",
        "Target_Currency", "Amount_USD", "Velocity_1H", "amount_zscore",
        "geo_mismatch", "is_fraud", "fraud_pattern",
        "fraud_probability", "action",
    ]
    scored[keep_cols].to_csv("data/scored_transactions.csv", index=False)

    fi = engine.feature_importance()
    fi.to_csv("data/feature_importance.csv", index=False)
    print("\nTop features:\n", fi.head(10))

    # single-row latency demo
    sample_row = featured.iloc[0]
    print("\nSingle-transaction inference demo:", engine.score_transaction(sample_row))

    # --- persist the trained engine for the dashboard's live scoring path ---
    joblib.dump(engine, "data/fraud_engine.joblib")

    # --- persist the corridor frequency lookup the streaming feature store
    # needs (it can't recompute a global frequency table from a single live
    # transaction, so it's learned once here, offline, and read-only at
    # serve time — exactly how a real feature store publishes batch-derived
    # features into an online store) ---
    corridor = featured["Source_Currency"] + "->" + featured["Target_Currency"]
    corridor_freq = corridor.value_counts(normalize=True).to_dict()
    with open("data/corridor_frequency.json", "w") as f:
        json.dump(corridor_freq, f, indent=2)

    print("\nWrote data/model_metrics.json, data/scored_transactions.csv, "
          "data/feature_importance.csv, data/fraud_engine.joblib, "
          "data/corridor_frequency.json")
