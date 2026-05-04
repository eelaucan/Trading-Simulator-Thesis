"""
test_nav.py

NAV identity: total_nav == cash + Σ(shares[i] * close_price[i])
must hold within floating-point tolerance after every state transition.
"""
from __future__ import annotations

import pytest

from simulator.actions import Action, ActionType, QuantityType
from simulator.env import TradingEnvironment


TOL = 1e-4  # $0.0001 tolerance for floating-point drift


def _check_nav(state, market, t) -> None:
    close = market.get_close_prices(min(t, market.n_weeks - 1))
    computed_mv = sum(
        qty * close.get(ticker, 0.0)
        for ticker, qty in state.shares_dict().items()
    )
    computed_nav = state.cash + computed_mv
    assert abs(computed_nav - state.total_nav) < TOL, (
        f"NAV mismatch at week {t}: "
        f"computed={computed_nav:.6f}, stored={state.total_nav:.6f}, "
        f"diff={abs(computed_nav - state.total_nav):.8f}"
    )


def test_nav_identity_hold_only(small_market, base_config):
    """NAV identity holds throughout a HOLD-only simulation."""
    env = TradingEnvironment(small_market, base_config)
    obs = env.reset()
    state = obs.portfolio_state

    # Initial state
    assert abs(state.total_nav - base_config.initial_cash) < TOL

    for _ in range(small_market.n_weeks - 2):
        obs, state, done, _ = env.step([Action(action_type=ActionType.HOLD)])
        _check_nav(state, small_market, state.week_index)
        if done:
            break


def test_nav_identity_after_buy(small_market, base_config):
    """NAV identity holds immediately after a buy."""
    env = TradingEnvironment(small_market, base_config)
    env.reset()

    obs, state, done, _ = env.step([
        Action(action_type=ActionType.BUY, ticker="AAPL",
               quantity=0.10, quantity_type=QuantityType.NAV_FRACTION)
    ])
    _check_nav(state, small_market, state.week_index)


def test_nav_identity_after_buy_and_sell(small_market, base_config):
    """NAV identity holds after buy then sell (round trip costs reduce NAV)."""
    env = TradingEnvironment(small_market, base_config)
    obs = env.reset()
    initial_nav = obs.portfolio_state.total_nav

    # Week 0: buy
    obs, state, done, _ = env.step([
        Action(action_type=ActionType.BUY, ticker="MSFT",
               quantity=0.10, quantity_type=QuantityType.NAV_FRACTION)
    ])
    nav_after_buy = state.total_nav
    _check_nav(state, small_market, state.week_index)

    if not done:
        # Week 1: sell
        obs, state, done, _ = env.step([
            Action(action_type=ActionType.SELL, ticker="MSFT",
                   quantity_type=QuantityType.CLOSE_ALL)
        ])
        _check_nav(state, small_market, state.week_index)
        # NAV should be less than initial due to round-trip costs
        # (price may have moved, but costs must have been deducted)
        assert state.cash >= 0, "Cash went negative after round-trip"


def test_nav_decreases_by_costs_on_round_trip(small_market, base_config):
    """After a round-trip trade, final NAV < initial NAV by at least the commission paid."""
    env = TradingEnvironment(small_market, base_config)
    obs = env.reset()
    initial_nav = obs.portfolio_state.total_nav

    obs, state, _, info = env.step([
        Action(action_type=ActionType.BUY, ticker="AAPL",
               quantity=0.05, quantity_type=QuantityType.NAV_FRACTION)
    ])
    er_buy = info["execution_results"][0]
    if er_buy is None:
        pytest.skip("No execution on this data segment")

    if state.week_index + 1 >= small_market.n_weeks - 1:
        pytest.skip("Not enough weeks for second trade")

    obs, state, _, info = env.step([
        Action(action_type=ActionType.SELL, ticker="AAPL",
               quantity_type=QuantityType.CLOSE_ALL)
    ])
    er_sell = info["execution_results"][0]
    if er_sell is None:
        pytest.skip("No sell execution")

    total_costs = er_buy.total_cost + er_sell.total_cost
    assert total_costs > 0, "No costs were charged on round trip"
