"""
test_no_lookahead.py

Guarantee: MarketReplay never returns data with _week_idx > t for any query.
"""
from __future__ import annotations


def test_get_history_never_returns_future(small_market):
    """get_history(t, large_lookback) must not include any week index > t."""
    market = small_market
    for t in range(market.n_weeks):
        hist = market.get_history(t, lookback=9999)
        assert "_week_idx" in hist.columns
        assert hist["_week_idx"].max() <= t, (
            f"At t={t}, get_history returned data up to week {hist['_week_idx'].max()}"
        )


def test_get_week_data_returns_only_t(small_market):
    """get_week_data(t) must return rows where _week_idx == t exactly."""
    market = small_market
    for t in range(market.n_weeks):
        week = market.get_week_data(t)
        assert (week["_week_idx"] == t).all(), (
            f"At t={t}, get_week_data returned rows with wrong _week_idx"
        )


def test_get_history_min_index(small_market):
    """get_history(t, k) lower bound is max(0, t-k)."""
    market = small_market
    t = 5
    hist = market.get_history(t, lookback=3)
    assert hist["_week_idx"].min() >= max(0, t - 3)
    assert hist["_week_idx"].max() <= t


def test_get_history_at_t0_has_only_week0(small_market):
    """At t=0 with any lookback, only week 0 is returned."""
    market = small_market
    hist = market.get_history(0, lookback=100)
    assert hist["_week_idx"].max() == 0


def test_observation_contains_no_future_data(small_market, base_config):
    """Observation price_history must have _week_idx <= obs.week_index."""
    from simulator.env import TradingEnvironment
    from simulator.actions import Action, ActionType

    env = TradingEnvironment(small_market, base_config)
    obs = env.reset()

    for _ in range(small_market.n_weeks - 2):
        assert obs.price_history["_week_idx"].max() <= obs.week_index, (
            f"Observation at week {obs.week_index} contains future price data"
        )
        obs, _, done, _ = env.step([Action(action_type=ActionType.HOLD)])
        if done:
            break
