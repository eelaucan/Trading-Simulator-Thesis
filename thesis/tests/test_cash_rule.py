"""
test_cash_rule.py

The cash rule must prevent post-trade cash from going below zero.
Buys that exceed available cash are clipped, not outright rejected.
"""
from __future__ import annotations

import pytest

from simulator.actions import Action, ActionType, QuantityType, ValidationOutcome
from simulator.env import TradingEnvironment
from simulator.execution import ExecutionEngine
from simulator.validator import ConstraintValidator


def test_buy_exceeding_cash_is_clipped(small_market, base_config):
    """A buy for more than available cash must be clipped, not rejected."""
    env = TradingEnvironment(small_market, base_config)
    env.reset()

    # Try to buy $200k worth of AAPL with only $100k cash
    action = Action(
        action_type=ActionType.BUY,
        ticker="AAPL",
        quantity=200_000.0,
        quantity_type=QuantityType.NOTIONAL_DOLLARS,
    )
    obs, state, done, info = env.step([action])

    vr = info["validation_results"][0]
    assert vr.outcome in (ValidationOutcome.CLIPPED, ValidationOutcome.ACCEPTED), (
        f"Expected CLIPPED or ACCEPTED, got {vr.outcome}"
    )
    # Cash must not be negative after execution
    assert state.cash >= -1e-6, f"Cash went negative: {state.cash}"


def test_hold_with_zero_cash_passes(small_market, base_config):
    """HOLD always passes regardless of cash level."""
    from simulator.state import PortfolioState
    from simulator.validator import ConstraintValidator
    from simulator.execution import ExecutionEngine
    from datetime import datetime

    cfg = base_config
    executor = ExecutionEngine(cfg)
    validator = ConstraintValidator(cfg, executor)

    state = PortfolioState.initial(0, datetime(2020, 1, 6), 0.0, cfg.ticker_universe)
    state = state.replace(cash=0.0)

    action = Action(action_type=ActionType.HOLD)
    vr = validator.validate(
        action=action,
        projected_state=state,
        market=small_market,
        t=0,
        accumulated_turnover_dollars=0.0,
        batch_start_nav=100_000.0,
    )
    assert vr.outcome == ValidationOutcome.ACCEPTED


def test_cash_never_negative_after_full_simulation(small_market, base_config):
    """After a full buy-and-hold run, cash must never go below zero."""
    env = TradingEnvironment(small_market, base_config)
    env.reset()

    # Buy all three tickers at start, then hold
    actions_week0 = [
        Action(action_type=ActionType.BUY, ticker="AAPL",
               quantity=0.30, quantity_type=QuantityType.NAV_FRACTION),
        Action(action_type=ActionType.BUY, ticker="MSFT",
               quantity=0.10, quantity_type=QuantityType.NAV_FRACTION),
    ]
    obs, state, done, info = env.step(actions_week0)
    assert state.cash >= -1e-6, f"Cash negative after buys: {state.cash}"

    while not done:
        obs, state, done, info = env.step([Action(action_type=ActionType.HOLD)])
        assert state.cash >= -1e-6, f"Cash went negative at week {state.week_index}: {state.cash}"
