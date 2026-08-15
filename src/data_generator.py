"""
data_generator.py
==================
Synthetic multi-currency transaction stream generator.

Design goals
------------
1. Realistic marginal distributions: remittance amounts are heavy-tailed
   (most transfers are small, a few are very large) -> modeled with a
   log-normal distribution per currency corridor rather than a normal
   distribution, which would produce an unrealistic number of negative or
   symmetric-around-the-mean amounts.
2. Realistic population structure: a Zipf-like distribution of transaction
   counts per user (most users transact once or twice, a small "power user"
   cohort transacts often) -> modeled by drawing a per-user transaction
   budget from a Pareto distribution.
3. A *hidden* ~2% fraud/AML layer injected via three archetypal patterns
   used by real transaction-monitoring systems:
     a. Structuring / smurfing: multiple transfers just under a reporting
        threshold ($10,000) from one user in a short window.
     b. Sudden high-volume spikes: an established "normal" user suddenly
        transacts 8-15x their historical average in a single transaction.
     c. Rapid multi-account access (device/credential fraud): a single
        Device_IP originates transactions from many distinct User_IDs
        within a short time window (mule network / account-takeover ring).
   These patterns are injected *after* the base population is generated so
   they are statistically "hidden" inside the legitimate distribution -
   exactly what a real model has to learn to separate.

The ground-truth `is_fraud` label is retained for supervised training and
evaluation, but is never used as a *feature*.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

CURRENCIES = ["USD", "GBP", "EUR", "INR", "PHP", "NGN", "BRL", "AUD"]

# Rough corridor popularity weights (Wise's real corridors skew heavily
# toward a handful of high-volume pairs) -> used to bias source/target draws.
CORRIDOR_WEIGHTS = {
    "USD": 0.28, "GBP": 0.18, "EUR": 0.20, "INR": 0.14,
    "PHP": 0.06, "NGN": 0.05, "BRL": 0.05, "AUD": 0.04,
}

# Currency -> plausible "home" country/IP block, used later to synthesize
# a geolocation-mismatch feature (a USD-source transaction whose device IP
# resolves to a country with no history in that currency is suspicious).
CURRENCY_IP_PREFIX = {
    "USD": "24.", "GBP": "81.", "EUR": "90.", "INR": "103.",
    "PHP": "112.", "NGN": "105.", "BRL": "177.", "AUD": "1.",
}


@dataclass
class GeneratorConfig:
    n_users: int = 8_000
    n_days: int = 30
    base_daily_txn_rate: float = 4_500          # legitimate txns/day, mean
    fraud_rate: float = 0.02                     # target fraction of ALL rows that are fraud-labeled
    random_seed: int = 42
    reporting_threshold_usd: float = 10_000.0    # structuring target
    start_date: datetime = field(default_factory=lambda: datetime(2026, 7, 1))


class TransactionStreamGenerator:
    """Generates a realistic, labeled, multi-currency transaction dataset."""

    def __init__(self, config: GeneratorConfig = GeneratorConfig()):
        self.cfg = config
        self.rng = np.random.default_rng(config.random_seed)

    # ------------------------------------------------------------------ #
    # Population setup
    # ------------------------------------------------------------------ #
    def _build_user_population(self) -> pd.DataFrame:
        """Assigns each user a home currency, a device IP, and a latent
        'typical transfer size' drawn log-normally -> this per-user mean
        is what lets us later compute a meaningful *personal* z-score
        (an amount that's normal for a whale is anomalous for a new user).
        """
        cfg = self.cfg
        user_ids = np.arange(1, cfg.n_users + 1)
        home_currency = self.rng.choice(
            list(CORRIDOR_WEIGHTS.keys()),
            size=cfg.n_users,
            p=list(CORRIDOR_WEIGHTS.values()),
        )
        # latent typical amount per user: log-normal, mean ~ $450, heavy tail
        typical_amount = self.rng.lognormal(mean=6.0, sigma=0.9, size=cfg.n_users)
        # activity budget: Pareto-distributed transaction counts (power users)
        activity_weight = (self.rng.pareto(a=2.0, size=cfg.n_users) + 1)

        device_ip = [
            f"{CURRENCY_IP_PREFIX[c]}{self.rng.integers(0,255)}.{self.rng.integers(0,255)}.{self.rng.integers(1,255)}"
            for c in home_currency
        ]

        return pd.DataFrame({
            "User_ID": user_ids,
            "home_currency": home_currency,
            "typical_amount_usd": typical_amount,
            "activity_weight": activity_weight,
            "home_device_ip": device_ip,
        })

    # ------------------------------------------------------------------ #
    # Legitimate transaction stream
    # ------------------------------------------------------------------ #
    def _generate_legitimate_stream(self, users: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        total_days = cfg.n_days
        expected_total = int(cfg.base_daily_txn_rate * total_days)

        # sample which user originates each transaction, weighted by activity
        p = (users["activity_weight"] / users["activity_weight"].sum()).values
        sampled_idx = self.rng.choice(len(users), size=expected_total, p=p)
        sampled_users = users.iloc[sampled_idx].reset_index(drop=True)

        # timestamps: uniform over the window, with a mild business-hours skew
        day_offset = self.rng.integers(0, total_days, size=expected_total)
        hour = self.rng.choice(
            np.arange(24),
            size=expected_total,
            p=_business_hour_profile(),
        )
        minute = self.rng.integers(0, 60, size=expected_total)
        second = self.rng.integers(0, 60, size=expected_total)
        timestamps = [
            cfg.start_date + timedelta(days=int(d), hours=int(h), minutes=int(m), seconds=int(s))
            for d, h, m, s in zip(day_offset, hour, minute, second)
        ]

        # amount: log-normal around each user's own typical amount
        amount = self.rng.lognormal(
            mean=np.log(sampled_users["typical_amount_usd"].values.clip(min=1)),
            sigma=0.35,
        )

        # target currency: usually different from source (home) currency
        target_currency = self.rng.choice(
            list(CORRIDOR_WEIGHTS.keys()), size=expected_total,
            p=list(CORRIDOR_WEIGHTS.values())
        )
        same_mask = target_currency == sampled_users["home_currency"].values
        # force a different target where they collided
        if same_mask.any():
            target_currency[same_mask] = self.rng.choice(
                list(CORRIDOR_WEIGHTS.keys()), size=same_mask.sum()
            )

        df = pd.DataFrame({
            "Timestamp": timestamps,
            "User_ID": sampled_users["User_ID"].values,
            "Source_Currency": sampled_users["home_currency"].values,
            "Target_Currency": target_currency,
            "Amount_USD": amount.round(2),
            "Device_IP": sampled_users["home_device_ip"].values,
            "is_fraud": 0,
            "fraud_pattern": "none",
        })
        return df

    # ------------------------------------------------------------------ #
    # Fraud injection
    # ------------------------------------------------------------------ #
    def _inject_structuring(self, df: pd.DataFrame, n_rings: int) -> pd.DataFrame:
        """Pattern A: structuring/smurfing. A user makes 3-6 transfers,
        each just under the reporting threshold, within a tight window
        (minutes to a couple hours) — classic AML evasion behavior."""
        cfg = self.cfg
        injected = []
        candidate_users = self.rng.choice(df["User_ID"].unique(), size=n_rings, replace=False)
        for uid in candidate_users:
            n_txn = self.rng.integers(3, 7)
            base_time = cfg.start_date + timedelta(
                days=int(self.rng.integers(0, cfg.n_days)),
                hours=int(self.rng.integers(0, 24)),
            )
            src_ccy = df.loc[df["User_ID"] == uid, "Source_Currency"].iloc[0]
            device = df.loc[df["User_ID"] == uid, "Device_IP"].iloc[0]
            for k in range(n_txn):
                amt = self.rng.uniform(0.85, 0.98) * cfg.reporting_threshold_usd
                ts = base_time + timedelta(minutes=int(self.rng.integers(2, 90)) * (k + 1))
                injected.append({
                    "Timestamp": ts,
                    "User_ID": uid,
                    "Source_Currency": src_ccy,
                    "Target_Currency": self.rng.choice(
                        [c for c in CURRENCIES if c != src_ccy]),
                    "Amount_USD": round(amt, 2),
                    "Device_IP": device,
                    "is_fraud": 1,
                    "fraud_pattern": "structuring",
                })
        return pd.DataFrame(injected)

    def _inject_volume_spikes(self, df: pd.DataFrame, users: pd.DataFrame,
                               n_spikes: int) -> pd.DataFrame:
        """Pattern B: an otherwise-normal, established user suddenly sends
        8-15x their historical typical amount in a single transaction —
        a classic account-takeover or laundering "cash-out" signature."""
        cfg = self.cfg
        injected = []
        sample_users = users.sample(n=n_spikes, random_state=int(self.rng.integers(0, 1e6)))
        for _, u in sample_users.iterrows():
            multiplier = self.rng.uniform(8, 15)
            amt = u["typical_amount_usd"] * multiplier
            ts = cfg.start_date + timedelta(
                days=int(self.rng.integers(0, cfg.n_days)),
                hours=int(self.rng.integers(0, 24)),
                minutes=int(self.rng.integers(0, 60)),
            )
            injected.append({
                "Timestamp": ts,
                "User_ID": u["User_ID"],
                "Source_Currency": u["home_currency"],
                "Target_Currency": self.rng.choice(
                    [c for c in CURRENCIES if c != u["home_currency"]]),
                "Amount_USD": round(amt, 2),
                "Device_IP": u["home_device_ip"],
                "is_fraud": 1,
                "fraud_pattern": "volume_spike",
            })
        return pd.DataFrame(injected)

    def _inject_device_rings(self, df: pd.DataFrame, n_rings: int) -> pd.DataFrame:
        """Pattern C: rapid multi-account logins / mule rings. A single
        Device_IP originates transactions tagged to 5-10 *distinct* user
        IDs within a short window — a strong signal of credential stuffing
        or a coordinated mule network sharing infrastructure."""
        cfg = self.cfg
        injected = []
        for _ in range(n_rings):
            ring_ip = f"185.{self.rng.integers(0,255)}.{self.rng.integers(0,255)}.{self.rng.integers(1,255)}"
            ring_users = self.rng.choice(df["User_ID"].unique(), size=self.rng.integers(5, 11), replace=False)
            base_time = cfg.start_date + timedelta(
                days=int(self.rng.integers(0, cfg.n_days)),
                hours=int(self.rng.integers(0, 24)),
            )
            for k, uid in enumerate(ring_users):
                src_ccy = df.loc[df["User_ID"] == uid, "Source_Currency"].iloc[0]
                ts = base_time + timedelta(minutes=int(self.rng.integers(1, 20)) * (k + 1))
                injected.append({
                    "Timestamp": ts,
                    "User_ID": uid,
                    "Source_Currency": src_ccy,
                    "Target_Currency": self.rng.choice(
                        [c for c in CURRENCIES if c != src_ccy]),
                    "Amount_USD": round(self.rng.uniform(200, 3000), 2),
                    "Device_IP": ring_ip,
                    "is_fraud": 1,
                    "fraud_pattern": "device_ring",
                })
        return pd.DataFrame(injected)

    # ------------------------------------------------------------------ #
    # Feature: rolling 1H velocity (count of txns by the same user in the
    # trailing 60 minutes) -> computed here so it ships as a raw input
    # column, exactly as it would arrive from a real streaming aggregator
    # (e.g. a Flink/Kafka Streams job), separate from the *training-time*
    # engineered features built in feature_engineering.py.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _add_velocity_1h(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("Timestamp").reset_index(drop=True)
        df["Velocity_1H"] = 0
        for uid, grp in df.groupby("User_ID"):
            idx = grp.index
            times = grp["Timestamp"].values.astype("datetime64[ns]")
            # sliding window count via searchsorted (O(n log n), vectorized)
            counts = np.searchsorted(times, times) - np.searchsorted(
                times, times - np.timedelta64(1, "h")
            )
            df.loc[idx, "Velocity_1H"] = counts
        return df

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def generate(self) -> pd.DataFrame:
        users = self._build_user_population()
        legit = self._generate_legitimate_stream(users)

        n_target_fraud = int(len(legit) * self.cfg.fraud_rate / (1 - self.cfg.fraud_rate))
        n_structuring = int(n_target_fraud * 0.4) // 5 + 1     # ~5 txns/ring
        n_spikes = int(n_target_fraud * 0.35)
        n_rings = int(n_target_fraud * 0.25) // 7 + 1          # ~7 txns/ring

        structuring = self._inject_structuring(legit, n_structuring)
        spikes = self._inject_volume_spikes(legit, users, n_spikes)
        rings = self._inject_device_rings(legit, n_rings)

        full = pd.concat([legit, structuring, spikes, rings], ignore_index=True)
        full = self._add_velocity_1h(full)
        full = full.sort_values("Timestamp").reset_index(drop=True)
        full.insert(0, "Transaction_ID", [f"TXN{100000+i}" for i in range(len(full))])

        actual_fraud_rate = full["is_fraud"].mean()
        print(f"[data_generator] Generated {len(full):,} transactions | "
              f"actual fraud rate = {actual_fraud_rate:.3%} (target {self.cfg.fraud_rate:.1%})")
        return full


def _business_hour_profile() -> np.ndarray:
    """Returns a 24-length probability vector skewed toward daytime hours,
    used to make transaction timing look human rather than uniform-random."""
    hours = np.arange(24)
    weights = np.exp(-0.5 * ((hours - 14) / 5.5) ** 2) + 0.15
    return weights / weights.sum()


if __name__ == "__main__":
    import os
    gen = TransactionStreamGenerator(GeneratorConfig())
    df = gen.generate()
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "transactions.csv")
    df.to_csv(out_path, index=False)
    print(f"[data_generator] wrote {out_path}")
    print(df["fraud_pattern"].value_counts())
