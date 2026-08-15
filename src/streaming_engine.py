"""
streaming_engine.py
=====================
This is what makes the dashboard genuinely "live" rather than a replay of a
pre-scored CSV: a **streaming feature store** that computes the exact same
behavioral features as `feature_engineering.py`, but incrementally — one
transaction at a time, using only bounded in-memory state (deques), the way
a real system (e.g. a Flink/Kafka-Streams job backing a feature store like
Feast or Tecton) would. There is no batch dataframe here; every feature is
derived from a rolling window maintained per user / per device as
transactions arrive.

Two classes:

  * `StreamingFeatureStore` — holds per-user and per-device rolling state
    and, given one new transaction, returns the feature vector for it in
    O(window size) time, then updates its own state. This is the
    "production" half.

  * `LiveTransactionSimulator` — generates one plausible transaction at a
    time (reusing the same population/behavior model as
    `data_generator.py`, at a much smaller scale suitable for a live
    UI), with the same three fraud archetypes injected probabilistically
    rather than deterministically pre-planted. This stands in for the
    "real" inbound transaction stream a production system would consume
    from Kafka/Kinesis.

Design note on honesty: this is still a *simulation* — there's no actual
message broker — but the feature computation is genuinely incremental and
streaming-correct, which is the part of "real-time systems" that's
actually interesting to demonstrate, and the part most portfolio projects
skip by secretly computing everything in a batch dataframe upfront.
"""
from __future__ import annotations

import random
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

REPORTING_THRESHOLD_USD = 10_000.0
CURRENCIES = ["USD", "GBP", "EUR", "INR", "PHP", "NGN", "BRL", "AUD"]
CORRIDOR_WEIGHTS = {
    "USD": 0.28, "GBP": 0.18, "EUR": 0.20, "INR": 0.14,
    "PHP": 0.06, "NGN": 0.05, "BRL": 0.05, "AUD": 0.04,
}
CURRENCY_IP_PREFIX = {
    "USD": "24.", "GBP": "81.", "EUR": "90.", "INR": "103.",
    "PHP": "112.", "NGN": "105.", "BRL": "177.", "AUD": "1.",
}


# ========================================================================
# Streaming feature store
# ========================================================================
class StreamingFeatureStore:
    """Maintains bounded, leak-free rolling state per user / device and
    computes the live feature vector for each incoming transaction on
    arrival, mirroring the feature set trained on in `fraud_model.py`.
    """

    def __init__(self, corridor_frequency: dict | None = None, history_days: int = 2):
        self.history_window = timedelta(days=history_days)
        # user_id -> deque[(timestamp, amount)]
        self.user_txns: dict[int, deque] = defaultdict(deque)
        # user_id -> deque[amount]  (bounded, for robust z-score)
        self.user_amount_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=50))
        # device_ip -> deque[(timestamp, user_id)]
        self.device_users: dict[str, deque] = defaultdict(deque)
        # user_id -> deque[(timestamp, device_ip)]
        self.user_devices: dict[int, deque] = defaultdict(deque)
        # static corridor frequency lookup, learned offline (from training data)
        self.corridor_frequency = corridor_frequency or {}
        self._global_corridor_default = (
            min(self.corridor_frequency.values()) if self.corridor_frequency else 0.01
        )

    def _trim(self, dq: deque, now: datetime, window: timedelta):
        while dq and (now - dq[0][0]) > window:
            dq.popleft()

    def process(self, txn: dict) -> dict:
        """Given a raw transaction dict, returns {**txn, **features} and
        updates internal rolling state. This is the one call a real
        low-latency scoring service would make per inbound message.
        """
        uid = txn["User_ID"]
        ip = txn["Device_IP"]
        ts = txn["Timestamp"]
        amt = txn["Amount_USD"]
        src = txn["Source_Currency"]

        one_hour = timedelta(hours=1)
        one_day = timedelta(hours=24)

        # ---- velocity / volume (strictly prior transactions only) ------
        hist = self.user_txns[uid]
        self._trim(hist, ts, one_day)
        hist_1h = [(t, a) for t, a in hist if ts - t <= one_hour]
        txn_count_1h = len(hist_1h)
        txn_sum_1h = sum(a for _, a in hist_1h)
        txn_count_24h = len(hist)
        txn_sum_24h = sum(a for _, a in hist)

        # ---- personalized robust z-score --------------------------------
        amt_hist = self.user_amount_history[uid]
        if len(amt_hist) >= 2:
            arr = np.array(amt_hist)
            median = np.median(arr)
            mad = np.median(np.abs(arr - median)) * 1.4826 + 1e-6
        else:
            median, mad = 500.0, 400.0  # cold-start prior, roughly global-population scale
        amount_zscore = (amt - median) / mad

        # ---- device / identity fan-out ----------------------------------
        dev_hist = self.device_users[ip]
        self._trim(dev_hist, ts, one_hour)
        distinct_users_per_ip_1h = len({u for _, u in dev_hist})

        ud_hist = self.user_devices[uid]
        self._trim(ud_hist, ts, one_hour)
        distinct_ips_per_user_1h = len({d for _, d in ud_hist})

        # ---- geolocation mismatch ----------------------------------------
        expected_prefix = CURRENCY_IP_PREFIX.get(src)
        geo_mismatch = int(expected_prefix is not None and not ip.startswith(expected_prefix))

        # ---- structuring proximity -----------------------------------------
        pct_of_threshold = amt / REPORTING_THRESHOLD_USD
        near_threshold_flag = int(0.80 * REPORTING_THRESHOLD_USD <= amt < REPORTING_THRESHOLD_USD)

        # ---- corridor frequency (static lookup learned offline) -----------
        corridor = f"{src}->{txn['Target_Currency']}"
        corridor_frequency = self.corridor_frequency.get(corridor, self._global_corridor_default)

        # ---- calendar --------------------------------------------------------
        hour_of_day = ts.hour
        is_night = int(hour_of_day < 6)
        day_of_week = ts.weekday()

        features = {
            "Amount_USD": amt,
            "Velocity_1H": txn_count_1h,
            "txn_count_1h": txn_count_1h,
            "txn_sum_1h": txn_sum_1h,
            "txn_count_24h": txn_count_24h,
            "txn_sum_24h": txn_sum_24h,
            "amount_zscore": amount_zscore,
            "distinct_users_per_ip_1h": distinct_users_per_ip_1h,
            "distinct_ips_per_user_1h": distinct_ips_per_user_1h,
            "geo_mismatch": geo_mismatch,
            "pct_of_reporting_threshold": pct_of_threshold,
            "near_threshold_flag": near_threshold_flag,
            "corridor_frequency": corridor_frequency,
            "hour_of_day": hour_of_day,
            "is_night": is_night,
            "day_of_week": day_of_week,
        }

        # ---- update state AFTER computing features (no self-leakage) ------
        hist.append((ts, amt))
        amt_hist.append(amt)
        dev_hist.append((ts, uid))
        ud_hist.append((ts, ip))

        return {**txn, **features}


# ========================================================================
# Live transaction simulator (stands in for a Kafka/Kinesis inbound stream)
# ========================================================================
@dataclass
class SimUser:
    user_id: int
    home_currency: str
    typical_amount: float
    device_ip: str


class LiveTransactionSimulator:
    """Generates one transaction at a time from a fixed synthetic
    population, injecting the same three fraud archetypes as
    `data_generator.py` but *probabilistically* and *live* — e.g. a
    "structuring burst" mode that fires several near-threshold
    transactions in a row for one user, then relaxes — rather than
    pre-planting a static fraud layer, so the live feed genuinely
    doesn't know what's coming next any more than a real system would.
    """

    def __init__(self, n_users: int = 400, seed: int | None = None,
                 fraud_probability: float = 0.03):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.fraud_probability = fraud_probability
        self.users: list[SimUser] = self._build_population(n_users)
        self.next_txn_id = 1
        self._active_burst: dict | None = None  # in-progress structuring/ring burst

    def _build_population(self, n_users: int) -> list[SimUser]:
        users = []
        for uid in range(1, n_users + 1):
            home = self.np_rng.choice(
                list(CORRIDOR_WEIGHTS.keys()), p=list(CORRIDOR_WEIGHTS.values())
            )
            typical = float(self.np_rng.lognormal(mean=6.0, sigma=0.9))
            ip = (f"{CURRENCY_IP_PREFIX[home]}{self.rng.randint(0,255)}."
                  f"{self.rng.randint(0,255)}.{self.rng.randint(1,255)}")
            users.append(SimUser(uid, home, typical, ip))
        return users

    def _random_target(self, exclude: str) -> str:
        choices = [c for c in CURRENCIES if c != exclude]
        return self.rng.choice(choices)

    def _next_id(self) -> str:
        tid = f"LIVE{self.next_txn_id:06d}"
        self.next_txn_id += 1
        return tid

    def _normal_txn(self, now: datetime) -> dict:
        u = self.rng.choice(self.users)
        amt = float(self.np_rng.lognormal(mean=np.log(max(u.typical_amount, 1)), sigma=0.35))
        return {
            "Transaction_ID": self._next_id(),
            "Timestamp": now,
            "User_ID": u.user_id,
            "Source_Currency": u.home_currency,
            "Target_Currency": self._random_target(u.home_currency),
            "Amount_USD": round(amt, 2),
            "Device_IP": u.device_ip,
            "is_fraud": 0,
            "fraud_pattern": "none",
        }

    def _start_structuring_burst(self, now: datetime) -> dict:
        u = self.rng.choice(self.users)
        remaining = self.rng.randint(2, 5)
        self._active_burst = {"type": "structuring", "user": u, "remaining": remaining}
        amt = self.rng.uniform(0.85, 0.98) * REPORTING_THRESHOLD_USD
        return {
            "Transaction_ID": self._next_id(), "Timestamp": now, "User_ID": u.user_id,
            "Source_Currency": u.home_currency, "Target_Currency": self._random_target(u.home_currency),
            "Amount_USD": round(amt, 2), "Device_IP": u.device_ip,
            "is_fraud": 1, "fraud_pattern": "structuring",
        }

    def _start_device_ring(self, now: datetime) -> dict:
        ring_ip = f"185.{self.rng.randint(0,255)}.{self.rng.randint(0,255)}.{self.rng.randint(1,255)}"
        ring_users = self.rng.sample(self.users, k=min(self.rng.randint(4, 7), len(self.users)))
        self._active_burst = {"type": "device_ring", "ip": ring_ip, "queue": ring_users[1:]}
        u = ring_users[0]
        return {
            "Transaction_ID": self._next_id(), "Timestamp": now, "User_ID": u.user_id,
            "Source_Currency": u.home_currency, "Target_Currency": self._random_target(u.home_currency),
            "Amount_USD": round(self.rng.uniform(200, 3000), 2), "Device_IP": ring_ip,
            "is_fraud": 1, "fraud_pattern": "device_ring",
        }

    def _volume_spike(self, now: datetime) -> dict:
        u = self.rng.choice(self.users)
        amt = u.typical_amount * self.rng.uniform(8, 15)
        return {
            "Transaction_ID": self._next_id(), "Timestamp": now, "User_ID": u.user_id,
            "Source_Currency": u.home_currency, "Target_Currency": self._random_target(u.home_currency),
            "Amount_USD": round(amt, 2), "Device_IP": u.device_ip,
            "is_fraud": 1, "fraud_pattern": "volume_spike",
        }

    def next_transaction(self, now: datetime | None = None) -> dict:
        """Returns the next single transaction in the live stream,
        continuing an in-progress fraud burst if one is active, otherwise
        rolling for a new normal / fraud event."""
        now = now or datetime.now()

        if self._active_burst is not None:
            burst = self._active_burst
            if burst["type"] == "structuring" and burst["remaining"] > 0:
                u = burst["user"]
                burst["remaining"] -= 1
                if burst["remaining"] <= 0:
                    self._active_burst = None
                amt = self.rng.uniform(0.85, 0.98) * REPORTING_THRESHOLD_USD
                return {
                    "Transaction_ID": self._next_id(), "Timestamp": now, "User_ID": u.user_id,
                    "Source_Currency": u.home_currency, "Target_Currency": self._random_target(u.home_currency),
                    "Amount_USD": round(amt, 2), "Device_IP": u.device_ip,
                    "is_fraud": 1, "fraud_pattern": "structuring",
                }
            if burst["type"] == "device_ring" and burst["queue"]:
                u = burst["queue"].pop(0)
                if not burst["queue"]:
                    self._active_burst = None
                return {
                    "Transaction_ID": self._next_id(), "Timestamp": now, "User_ID": u.user_id,
                    "Source_Currency": u.home_currency, "Target_Currency": self._random_target(u.home_currency),
                    "Amount_USD": round(self.rng.uniform(200, 3000), 2), "Device_IP": burst["ip"],
                    "is_fraud": 1, "fraud_pattern": "device_ring",
                }
            self._active_burst = None  # safety fallback

        roll = self.rng.random()
        if roll < self.fraud_probability * 0.4:
            return self._start_structuring_burst(now)
        elif roll < self.fraud_probability * 0.7:
            return self._volume_spike(now)
        elif roll < self.fraud_probability:
            return self._start_device_ring(now)
        return self._normal_txn(now)

    def next_batch(self, n: int, start_time: datetime | None = None,
                    seconds_between: float = 0.8) -> list[dict]:
        start_time = start_time or datetime.now()
        out = []
        for i in range(n):
            ts = start_time + timedelta(seconds=i * seconds_between)
            out.append(self.next_transaction(ts))
        return out
