"""
test_turnover.py

Weekly turnover = gross_traded / batch_start_nav must stay <= turnover_cap.
The denominator is batch_start_nav (fixed at the start of each step).
"""
from __future__ import annotations

from simulator.actions import Action, ActionType, QuantityType, ValidationOutcome
from simulator.env import TradingEnvironment


def test_turnover_cap_clips_oversized_trade(small_market, base_config):
    """A single trade that would exceed the turnover cap must be clipped."""
    env = TradingEnvironment(small_market, base_config)
    env.reset()

    # Attempt to buy 50% of NAV in one step (cap is 25%)
    action = Action(
        action_type=ActionType.BUY,
        ticker="AAPL",
        quantity=0.50,
        quantity_type=QuantityType.NAV_FRACTION,
    )
    obs, state, done, info = env.step([action])

    vr = info["validation_results"][0]
    # Should be clipped (or accepted if other rules are more restrictive)
    # The key check is weekly_turnover <= cap + small tolerance
    assert state.weekly_turnover <= base_config.turnover_cap + 0.01, (
        f"weekly_turnover {state.weekly_turnover:.4f} exceeds cap {base_config.turnover_cap}"
    )


def test_turnover_accumulates_across_batch(small_market, base_config):
    """Multiple buys in the same batch accumulate toward the turnover cap."""
    env = TradingEnvironment(small_market, base_config)
    env.reset()

    # Each 15% — together they would be 30%, over the 25% cap
    actions = [
        Action(action_type=ActionType.BUY, ticker="AAPL",
               quantity=0.15, quantity_type=QuantityType.NAV_FRACTION),
        Action(action_type=ActionType.BUY, ticker="MSFT",
               quantity=0.15, quantity_type=QuantityType.NAV_FRACTION),
    ]
    obs, state, done, info = env.step(actions)

    assert state.weekly_turnover <= base_config.turnover_cap + 0.01, (
        f"weekly_turnover {state.weekly_turnover:.4f} exceeds cap {base_config.turnover_cap}"
    )


def test_turnover_uses_batch_start_nav(small_market, base_config):
    """
    Turnover denominator must be batch_start_nav, not changing NAV.
    We verify by checking that weekly_turnover == gross_traded / batch_start_nav.
    """
    env = TradingEnvironment(small_market, base_config)
    obs = env.reset()
    batch_start_nav = obs.portfolio_state.total_nav

    action = Action(
        action_type=ActionType.BUY,
        ticker="AAPL",
        quantity=0.10,
        quantity_type=QuantityType.NAV_FRACTION,
    )
    obs, state, done, info = env.step([action])

    er = info["execution_results"][0]
    if er is not None:
        expected_turnover = er.trade_value / batch_start_nav
        assert abs(state.weekly_turnover - expected_turnover) < 0.01, (
            f"Turnover {state.weekly_turnover:.4f} != trade_value/batch_start_nav {expected_turnover:.4f}"
        )
