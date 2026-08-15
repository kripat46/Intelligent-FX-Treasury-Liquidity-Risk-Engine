"""
tests/test_liquidity_optimizer.py
====================================
Tests the newsvendor optimization math directly: the closed-form and
numerical solutions must agree, the reorder point must decompose
correctly, and the dynamic fee curve must be monotonic and bounded.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from liquidity_optimizer import (  # noqa: E402
    LiquidityOptimizer,
    LivePoolBook,
    PoolParameters,
)


def _sample_pool(currency="USD", current_level=500_000.0) -> PoolParameters:
    return PoolParameters(
        currency=currency,
        daily_outflow_mean=100_000.0,
        daily_outflow_std=30_000.0,
        lead_time_days=1.5,
        lead_time_std_days=0.4,
        holding_cost_per_unit_per_day=0.045 / 365,
        stockout_cost_per_unit=0.01,
        current_level=current_level,
    )


def test_closed_form_and_numerical_optimum_agree():
    optimizer = LiquidityOptimizer([_sample_pool()])
    result = optimizer.optimize_pool("USD")
    rel_diff = abs(result.safety_stock - result.numerical_safety_stock) / result.safety_stock
    assert rel_diff < 0.02, "closed-form and numerical safety stock diverged"


def test_reorder_point_equals_demand_plus_safety_stock():
    optimizer = LiquidityOptimizer([_sample_pool()])
    result = optimizer.optimize_pool("USD")
    expected_rop = result.demand_during_lead_time_mean + result.safety_stock
    assert abs(result.reorder_point - expected_rop) < 1.0


def test_needs_rebalancing_flag_matches_level_vs_reorder_point():
    low = LiquidityOptimizer([_sample_pool(current_level=1.0)]).optimize_pool("USD")
    high = LiquidityOptimizer([_sample_pool(current_level=10_000_000.0)]).optimize_pool("USD")
    assert low.needs_rebalancing is True
    assert high.needs_rebalancing is False


def test_fee_multiplier_is_bounded_and_monotonic_in_scarcity():
    optimizer = LiquidityOptimizer([_sample_pool()])
    fees = [
        optimizer._fee_multiplier(scarcity_ratio=s)
        for s in [3.0, 1.0, 0.0, -1.0, -3.0]
    ]
    assert all(1.0 <= f <= 2.5 for f in fees)
    # more scarce (more negative ratio) -> fee should not decrease
    assert fees == sorted(fees), "fee multiplier must be non-decreasing as scarcity increases"


def test_live_pool_book_depletes_and_replenishes():
    pool = _sample_pool(currency="INR", current_level=50_000.0)  # start already scarce
    book = LivePoolBook([pool], replenishment_lead_ticks=2)

    snap0 = book.tick({"INR": 0.0})
    assert bool(snap0.iloc[0]["needs_rebalancing"]) is True

    # tick through the replenishment lead time with no further outflow
    book.tick({"INR": 0.0})
    snap2 = book.tick({"INR": 0.0})
    # after replenishment_lead_ticks ticks, level should have jumped up to
    # the order-up-to target and no longer need rebalancing
    assert snap2.iloc[0]["current_level"] > snap0.iloc[0]["current_level"]
