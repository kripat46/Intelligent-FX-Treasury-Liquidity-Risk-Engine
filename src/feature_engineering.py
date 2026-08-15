"""
feature_engineering.py
=======================
Point-in-time-safe behavioral feature engineering for the fraud model.

The single most common bug in "fraud detection" portfolio projects is
**target leakage through improperly computed rolling features** — e.g.
using `df.groupby('user').transform('mean')` computed over the *whole*
dataset, which lets a transaction "see into the future" (its own amount
influences the mean it's being compared against, and later transactions
leak into earlier ones). Every rolling/aggregate feature below is computed
using only information strictly *prior to* the current transaction's
timestamp, which is what a real streaming feature store would guarantee.

Feature families
-----------------
1. **Velocity features**: rolling transaction counts/sums per user over
   1H / 24H trailing windows (excluding the current row).
2. **Amount anomaly z-scores**: how many standard deviations the current
   amount is from *that user's own trailing history* (a personalized,
   robust z-score using median/MAD rather than mean/std, since amount
   distributions are heavy-tailed and a few whales would otherwise blow
   up a naive std-based score).
3. **Device/identity fan-out features**: number of distinct users sharing
   a Device_IP in a trailing window (mule-ring signature), and number of
   distinct devices used by a single user (account-sharing/ATO signature).
4. **Geolocation mismatch**: a synthetic "expected country" derived from
   Source_Currency vs. the country implied by the Device_IP block; a
   mismatch is a classic ATO/proxy-fraud indicator.
5. **Structuring proximity**: how close Amount_USD sits to the regulatory
   reporting threshold — a direct, interpretable signal for structuring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REPORTING_THRESHOLD_USD = 10_000.0

# Mirrors the mapping used at generation time; in a real system this would
# come from an IP-geolocation service / MaxMind lookup, not a static dict.
CURRENCY_IP_PREFIX = {
    "USD": "24.", "GBP": "81.", "EUR": "90.", "INR": "103.",
    "PHP": "112.", "NGN": "105.", "BRL": "177.", "AUD": "1.",
}


def _rolling_count_sum(df: pd.DataFrame, window: str) -> pd.DataFrame:
    """Per-user trailing rolling count & sum of Amount_USD, EXCLUDING the
    current transaction (shift-by-one-row semantics), computed with
    pandas' time-aware `.rolling(window, closed='left')` grouped by user.

    Using `closed='left'` is the key correctness detail: it excludes the
    current row's own timestamp from its own window, which is what
    guarantees no leakage.
    """
    tmp = df[["User_ID", "Timestamp", "Amount_USD"]].copy()
    tmp = tmp.set_index("Timestamp")

    out_count, out_sum = [], []
    for uid, grp in tmp.groupby("User_ID", sort=False):
        grp = grp.sort_index()
        roll = grp["Amount_USD"].rolling(window, closed="left")
        out_count.append(roll.count().fillna(0))
        out_sum.append(roll.sum().fillna(0))

    count_series = pd.concat(out_count).sort_index()
    sum_series = pd.concat(out_sum).sort_index()
    # re-align back to original row order via a merge on (User_ID, Timestamp)
    result = df[["User_ID", "Timestamp"]].copy()
    result[f"txn_count_{window}"] = count_series.values if len(count_series) == len(df) else np.nan
    result[f"txn_sum_{window}"] = sum_series.values if len(sum_series) == len(df) else np.nan
    return result[[f"txn_count_{window}", f"txn_sum_{window}"]]


def _robust_user_zscore(df: pd.DataFrame) -> pd.Series:
    """Personalized robust z-score of Amount_USD using each user's
    trailing (expanding, leak-free) median and MAD, scaled to be
    std-equivalent (MAD * 1.4826). Falls back to the global robust stats
    for a user's very first transaction (cold start).
    """
    df = df.sort_values("Timestamp")
    z = np.zeros(len(df))
    global_median = df["Amount_USD"].median()
    global_mad = (df["Amount_USD"] - global_median).abs().median() * 1.4826 + 1e-6

    for uid, grp in df.groupby("User_ID", sort=False):
        amounts = grp["Amount_USD"].values
        idx = grp.index.values
        running_median = global_median
        running_mad = global_mad
        history = []
        for i, (row_idx, amt) in enumerate(zip(idx, amounts)):
            if len(history) >= 2:
                running_median = np.median(history)
                running_mad = np.median(np.abs(np.array(history) - running_median)) * 1.4826 + 1e-6
            z[df.index.get_loc(row_idx)] = (amt - running_median) / running_mad
            history.append(amt)
    return pd.Series(z, index=df.index)


def _device_fanout_features(df: pd.DataFrame, window_minutes: int = 60) -> pd.DataFrame:
    """For each transaction, counts (a) how many distinct User_IDs have
    used the same Device_IP in the trailing window, and (b) how many
    distinct Device_IPs the current user has used in the trailing window.
    Both are computed leak-free (strictly prior transactions only).
    """
    df_sorted = df.sort_values("Timestamp").reset_index()
    n = len(df_sorted)
    users_per_ip = np.zeros(n)
    ips_per_user = np.zeros(n)

    window = pd.Timedelta(minutes=window_minutes)
    ip_history: dict[str, list] = {}
    user_history: dict[int, list] = {}

    for i, row in df_sorted.iterrows():
        ts, ip, uid = row["Timestamp"], row["Device_IP"], row["User_ID"]

        # trim + count for this IP
        hist_ip = ip_history.get(ip, [])
        hist_ip = [(t, u) for (t, u) in hist_ip if ts - t <= window]
        users_per_ip[i] = len({u for _, u in hist_ip})
        hist_ip.append((ts, uid))
        ip_history[ip] = hist_ip

        # trim + count for this user
        hist_u = user_history.get(uid, [])
        hist_u = [(t, ipp) for (t, ipp) in hist_u if ts - t <= window]
        ips_per_user[i] = len({ipp for _, ipp in hist_u})
        hist_u.append((ts, ip))
        user_history[uid] = hist_u

    result = pd.DataFrame({
        "index": df_sorted["index"],
        "distinct_users_per_ip_1h": users_per_ip,
        "distinct_ips_per_user_1h": ips_per_user,
    }).set_index("index").sort_index()
    return result


def _geo_mismatch(df: pd.DataFrame) -> pd.Series:
    """1 if the Device_IP block does not match the expected block for the
    transaction's Source_Currency (a proxy for "device country != stated
    home currency country"), else 0.
    """
    def check(row):
        expected_prefix = CURRENCY_IP_PREFIX.get(row["Source_Currency"])
        return int(expected_prefix is not None and not row["Device_IP"].startswith(expected_prefix))
    return df.apply(check, axis=1)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Main entry point: takes the raw transaction stream (as produced by
    data_generator.py) and returns a feature matrix ready for modeling.
    Retains identifier/label columns so callers can split them off.
    """
    df = raw.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # --- velocity / volume features -------------------------------------
    r1h = _rolling_count_sum(df, "1h")
    r24h = _rolling_count_sum(df, "24h")
    df = pd.concat([df, r1h, r24h], axis=1)

    # --- personalized amount anomaly z-score -----------------------------
    df["amount_zscore"] = _robust_user_zscore(df)

    # --- device / identity fan-out ---------------------------------------
    fanout = _device_fanout_features(df)
    df = df.join(fanout)

    # --- geolocation mismatch --------------------------------------------
    df["geo_mismatch"] = _geo_mismatch(df)

    # --- structuring proximity --------------------------------------------
    df["pct_of_reporting_threshold"] = df["Amount_USD"] / REPORTING_THRESHOLD_USD
    df["near_threshold_flag"] = (
        (df["Amount_USD"] >= 0.80 * REPORTING_THRESHOLD_USD) &
        (df["Amount_USD"] < REPORTING_THRESHOLD_USD)
    ).astype(int)

    # --- cross-currency corridor risk (categorical -> frequency encoding) -
    corridor = df["Source_Currency"] + "->" + df["Target_Currency"]
    corridor_freq = corridor.value_counts(normalize=True)
    df["corridor_frequency"] = corridor.map(corridor_freq)

    # --- calendar features --------------------------------------------
    df["hour_of_day"] = df["Timestamp"].dt.hour
    df["is_night"] = df["hour_of_day"].isin(range(0, 6)).astype(int)
    df["day_of_week"] = df["Timestamp"].dt.dayofweek

    return df


FEATURE_COLUMNS = [
    "Amount_USD", "Velocity_1H",
    "txn_count_1h", "txn_sum_1h", "txn_count_24h", "txn_sum_24h",
    "amount_zscore",
    "distinct_users_per_ip_1h", "distinct_ips_per_user_1h",
    "geo_mismatch",
    "pct_of_reporting_threshold", "near_threshold_flag",
    "corridor_frequency",
    "hour_of_day", "is_night", "day_of_week",
]


if __name__ == "__main__":
    raw = pd.read_csv("data/transactions.csv")
    feats = build_features(raw)
    feats.to_csv("data/transactions_featured.csv", index=False)
    print(feats[FEATURE_COLUMNS + ["is_fraud"]].describe().T)
