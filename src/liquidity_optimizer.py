"""
liquidity_optimizer.py
========================
Multi-currency treasury float optimization engine.

Problem framing
----------------
Each local-currency pool (e.g. the platform's pre-funded INR account that
pays out INR-denominated transfers) behaves like an inventory of a
perishable resource:

  * Daily net outflow (demand) is stochastic, not deterministic — modeled
    here as Normal(mu_d, sigma_d), estimated from the historical outflow
    implied by the transaction stream.
  * Replenishment (topping the pool back up via an FX conversion / wire)
    takes a **lead time** L (days) that is *also* uncertain.
  * Holding excess cash in a pool has an opportunity cost (cost of capital
    — that cash isn't earning yield or being deployed elsewhere).
  * Running a pool dry (a "stockout") delays customer transfers, which the
    business models as a cost per unit of currency short (SLA penalties,
    support burden, and reputational/churn risk).

This is the classical **newsvendor / (s, S) inventory problem**, and it has
a well-known closed-form optimum. We implement it two ways so the two
independently confirm each other:

  1. **Closed-form critical-ratio solution** (the textbook newsvendor
     result): the cost-minimizing service level equals
     `Cu / (Cu + Co)` where Cu = underage (stockout) cost per unit,
     Co = overage (holding) cost per unit. Reorder point R and safety
     stock SS follow directly from the demand-during-lead-time
     distribution's quantile at that service level.
  2. **Direct numerical optimization** via `scipy.optimize.minimize_scalar`,
     minimizing the total expected cost function
     `E[Cost(SS)] = Co * SS + Cu * E[shortfall | SS]`
     over safety stock SS, where the expected shortfall uses the Normal
     loss function. This is the "advanced heuristic / optimization model"
     requested — a genuine numerical optimization, not just a formula
     lookup — and its optimum should match (1) up to solver tolerance,
     which the module asserts.

Dynamic pricing
-----------------
The transfer fee for a given currency corridor is adjusted by a **scarcity
multiplier**: a logistic function of how far the current pool level sits
below its reorder point, expressed in safety-stock units. This:
  * leaves fees untouched while the pool is healthy (level >> reorder point),
  * ramps fees up smoothly (not as a step function, which would create
    exploitable fee-arbitrage edges) as the pool approaches its reorder
    point,
  * caps at a maximum surcharge multiplier so pricing stays within a
    regulator-defensible band.
This both raises revenue to fund emergency replenishment and organically
throttles outflow demand on the scarce currency exactly when the platform
needs it throttled — a self-correcting mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar


@dataclass
class PoolParameters:
    """Estimated/assumed parameters for a single currency float pool."""
    currency: str
    daily_outflow_mean: float          # mu_d: mean daily net outflow, in pool currency
    daily_outflow_std: float           # sigma_d: std of daily net outflow
    lead_time_days: float              # L: mean replenishment lead time
    lead_time_std_days: float          # sigma_L: uncertainty in lead time
    holding_cost_per_unit_per_day: float  # Co: opportunity cost of idle cash (e.g. daily cost of capital)
    stockout_cost_per_unit: float      # Cu: cost of a unit short (SLA penalty / churn cost)
    current_level: float               # current pool balance


@dataclass
class OptimizationResult:
    currency: str
    demand_during_lead_time_mean: float
    demand_during_lead_time_std: float
    critical_ratio: float
    service_level: float
    z_score: float
    safety_stock: float
    reorder_point: float
    numerical_safety_stock: float      # from direct scipy optimization, should ~= safety_stock
    expected_total_cost_at_optimum: float
    current_level: float
    scarcity_ratio: float              # (level - reorder_point) / safety_stock ; <0 means below ROP
    fee_multiplier: float
    needs_rebalancing: bool


class LiquidityOptimizer:
    """Solves the (s, S)-style newsvendor problem per currency pool and
    derives a dynamic pricing multiplier from pool scarcity."""

    def __init__(self, pools: list[PoolParameters]):
        self.pools = {p.currency: p for p in pools}

    # ------------------------------------------------------------------ #
    # Demand-during-lead-time distribution
    # ------------------------------------------------------------------ #
    @staticmethod
    def _demand_during_lead_time(p: PoolParameters) -> tuple[float, float]:
        """Combines daily demand uncertainty AND lead-time uncertainty into
        the distribution of *total demand over the replenishment window*.

        mu_LT    = L * mu_d
        sigma_LT = sqrt( L * sigma_d^2  +  mu_d^2 * sigma_L^2 )

        The second term is the standard inventory-theory correction for
        stochastic lead time: even if daily demand were perfectly known,
        an uncertain lead time alone injects variance proportional to
        (mean demand)^2 * (lead-time variance). Omitting it is a common
        analytical error that understates real risk.
        """
        mu_lt = p.lead_time_days * p.daily_outflow_mean
        var_lt = (
            p.lead_time_days * p.daily_outflow_std ** 2
            + (p.daily_outflow_mean ** 2) * (p.lead_time_std_days ** 2)
        )
        return mu_lt, float(np.sqrt(var_lt))

    # ------------------------------------------------------------------ #
    # Closed-form newsvendor solution
    # ------------------------------------------------------------------ #
    @staticmethod
    def _closed_form(mu_lt: float, sigma_lt: float, Co: float, Cu: float) -> tuple[float, float, float, float]:
        critical_ratio = Cu / (Cu + Co)
        z = stats.norm.ppf(critical_ratio)
        safety_stock = z * sigma_lt
        reorder_point = mu_lt + safety_stock
        return critical_ratio, critical_ratio, z, max(safety_stock, 0.0), reorder_point  # noqa

    # ------------------------------------------------------------------ #
    # Direct numerical optimization (confirms the closed form)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _numerical_optimum(mu_lt: float, sigma_lt: float, Co: float, Cu: float) -> tuple[float, float]:
        """Minimizes E[Cost(SS)] = Co*SS + Cu * sigma_LT * L(z), where
        L(z) is the standard normal loss function
        L(z) = phi(z) - z*(1 - Phi(z)), and SS = z * sigma_LT.
        This is solved directly over SS via bounded scalar minimization,
        independent of the critical-ratio shortcut, as a numerical check.
        """
        def expected_cost(ss: float) -> float:
            if sigma_lt <= 0:
                return Co * ss
            z = ss / sigma_lt
            loss = stats.norm.pdf(z) - z * (1 - stats.norm.cdf(z))
            expected_shortfall = sigma_lt * loss
            return Co * ss + Cu * expected_shortfall

        res = minimize_scalar(
            expected_cost, bounds=(0, 8 * sigma_lt + 1e-6), method="bounded"
        )
        return float(res.x), float(res.fun)

    # ------------------------------------------------------------------ #
    # Dynamic pricing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fee_multiplier(scarcity_ratio: float, max_multiplier: float = 2.5,
                         steepness: float = 2.0) -> float:
        """Logistic scarcity-to-fee mapping.

        scarcity_ratio = (current_level - reorder_point) / safety_stock
          > 0  : pool comfortably above its reorder point -> multiplier -> 1.0
          = 0  : pool exactly at the reorder point -> multiplier at curve midpoint
          << 0 : pool deep into/through safety stock -> multiplier -> max_multiplier

        Implemented as a logistic curve in `-scarcity_ratio` so it's smooth
        (differentiable, no fee-arbitrage step edges) and bounded in
        [1.0, max_multiplier].
        """
        x = -scarcity_ratio * steepness
        sigmoid = 1 / (1 + np.exp(-x))
        return float(1.0 + (max_multiplier - 1.0) * sigmoid)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def optimize_pool(self, currency: str) -> OptimizationResult:
        p = self.pools[currency]
        mu_lt, sigma_lt = self._demand_during_lead_time(p)

        critical_ratio, service_level, z, safety_stock, reorder_point = self._closed_form(
            mu_lt, sigma_lt, p.holding_cost_per_unit_per_day, p.stockout_cost_per_unit
        )
        numerical_ss, expected_cost = self._numerical_optimum(
            mu_lt, sigma_lt, p.holding_cost_per_unit_per_day, p.stockout_cost_per_unit
        )

        scarcity_ratio = (
            (p.current_level - reorder_point) / safety_stock if safety_stock > 0 else 0.0
        )
        fee_multiplier = self._fee_multiplier(scarcity_ratio)

        return OptimizationResult(
            currency=currency,
            demand_during_lead_time_mean=round(mu_lt, 2),
            demand_during_lead_time_std=round(sigma_lt, 2),
            critical_ratio=round(critical_ratio, 4),
            service_level=round(service_level, 4),
            z_score=round(z, 4),
            safety_stock=round(safety_stock, 2),
            reorder_point=round(reorder_point, 2),
            numerical_safety_stock=round(numerical_ss, 2),
            expected_total_cost_at_optimum=round(expected_cost, 2),
            current_level=round(p.current_level, 2),
            scarcity_ratio=round(scarcity_ratio, 3),
            fee_multiplier=round(fee_multiplier, 3),
            needs_rebalancing=bool(p.current_level < reorder_point),
        )

    def optimize_all(self) -> list[OptimizationResult]:
        return [self.optimize_pool(c) for c in self.pools]

    def to_dataframe(self) -> pd.DataFrame:
        results = self.optimize_all()
        return pd.DataFrame([r.__dict__ for r in results])


@dataclass
class LivePool:
    """Wraps a PoolParameters with the extra state needed to simulate an
    (s, S) inventory policy evolving tick-by-tick in a live UI: once the
    pool's current_level drops to/below its reorder point (s), a
    replenishment order is placed; it arrives after a countdown standing
    in for the lead time, at which point the level is restored to the
    order-up-to level S = reorder_point + safety_stock (the standard
    (s, S) target — order enough to cover expected demand during the next
    lead time plus the safety buffer).

    The countdown is expressed in "ticks" (UI refreshes) rather than
    literal days, since a live demo needs replenishment to be observable
    within a session — this is a presentation-layer simplification of the
    same lead-time concept used in the day-scale optimizer, not a
    different model.
    """
    params: PoolParameters
    replenishment_lead_ticks: int = 6
    _pending_ticks: int = 0

    def apply_outflow(self, amount: float) -> None:
        self.params.current_level = max(0.0, self.params.current_level - amount)

    def tick(self, optimizer: "LiquidityOptimizer") -> "OptimizationResult":
        result = optimizer.optimize_pool(self.params.currency)
        if result.needs_rebalancing and self._pending_ticks == 0:
            self._pending_ticks = self.replenishment_lead_ticks
        if self._pending_ticks > 0:
            self._pending_ticks -= 1
            if self._pending_ticks == 0:
                target_level = result.reorder_point + result.safety_stock  # order-up-to S
                self.params.current_level = target_level
                result = optimizer.optimize_pool(self.params.currency)
        return result

    @property
    def replenishment_pending(self) -> bool:
        return self._pending_ticks > 0


class LivePoolBook:
    """Manages a collection of LivePool objects and evolves all of them
    together each tick, given a dict of {currency: outflow_amount} drawn
    from the latest batch of live transactions."""

    def __init__(self, pool_params: list[PoolParameters], replenishment_lead_ticks: int = 6):
        self.pools = {
            p.currency: LivePool(params=p, replenishment_lead_ticks=replenishment_lead_ticks)
            for p in pool_params
        }

    def tick(self, outflows: dict[str, float]) -> pd.DataFrame:
        for currency, amount in outflows.items():
            if currency in self.pools and amount > 0:
                self.pools[currency].apply_outflow(amount)

        optimizer = LiquidityOptimizer([lp.params for lp in self.pools.values()])
        rows = []
        for currency, lp in self.pools.items():
            result = lp.tick(optimizer)
            row = result.__dict__.copy()
            row["replenishment_pending"] = lp.replenishment_pending
            rows.append(row)
        return pd.DataFrame(rows)


def estimate_pool_parameters_from_transactions(
    df: pd.DataFrame, currency: str,
    holding_cost_apr: float = 0.045,      # annualized cost of capital assumption
    stockout_cost_per_unit: float = 0.01, # assumed cost per unit-currency short:
                                           # ~1% "emergency premium" (spread on a rushed
                                           # spot FX conversion + amortized SLA penalty).
                                           # Chosen so Cu/Co lands in a realistic 98-99%
                                           # critical-infrastructure service-level band
                                           # rather than a degenerate ~100%.
    lead_time_days: float = 1.5,
    lead_time_std_days: float = 0.4,
    starting_level_days_of_cover: float = 5.0,
) -> PoolParameters:
    """Derives PoolParameters from the observed transaction stream: any
    transaction whose Source_Currency == currency draws DOWN that pool
    (an outbound payout in that currency), aggregated to a daily outflow
    series, from which we estimate mu_d and sigma_d empirically rather
    than assuming them — grounding the optimization in the same data the
    fraud model uses.
    """
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["date"] = df["Timestamp"].dt.date

    outflow = (
        df[df["Source_Currency"] == currency]
        .groupby("date")["Amount_USD"]
        .sum()
    )
    # reindex to include zero-outflow days so mean/std reflect true daily variability
    full_range = pd.date_range(df["date"].min(), df["date"].max())
    outflow = outflow.reindex(full_range.date, fill_value=0.0)

    mu_d = float(outflow.mean())
    sigma_d = float(outflow.std(ddof=1))
    daily_holding_cost = holding_cost_apr / 365.0

    return PoolParameters(
        currency=currency,
        daily_outflow_mean=mu_d,
        daily_outflow_std=sigma_d,
        lead_time_days=lead_time_days,
        lead_time_std_days=lead_time_std_days,
        holding_cost_per_unit_per_day=daily_holding_cost,
        stockout_cost_per_unit=stockout_cost_per_unit,
        current_level=mu_d * starting_level_days_of_cover,
    )


if __name__ == "__main__":
    df = pd.read_csv("data/transactions.csv")
    pools = [
        estimate_pool_parameters_from_transactions(df, ccy)
        for ccy in ["USD", "INR", "GBP", "EUR", "PHP", "NGN", "BRL", "AUD"]
    ]

    # Deliberately understock a couple of pools to demonstrate the alerting
    # path in the dashboard (in a live system this would just be the real
    # current balance from the ledger).
    for p in pools:
        if p.currency in ("INR", "NGN"):
            p.current_level *= 0.35

    optimizer = LiquidityOptimizer(pools)
    results_df = optimizer.to_dataframe()
    print(results_df.to_string(index=False))

    # sanity check: closed-form and numerical safety stock should agree closely.
    # Compared in RELATIVE terms since pool sizes span ~1e5-1e6 units; solver
    # tolerance on minimize_scalar naturally yields absolute gaps at that scale.
    rel_diffs = (
        (results_df["safety_stock"] - results_df["numerical_safety_stock"]).abs()
        / results_df["safety_stock"]
    )
    print(f"\nMax relative |closed_form_SS - numerical_SS| across pools: {rel_diffs.max():.6%}")
    assert rel_diffs.max() < 0.01, "Closed-form and numerical optimum diverged unexpectedly"

    results_df.to_json("data/liquidity_state.json", orient="records", indent=2)
    print("\nWrote data/liquidity_state.json")
