"""
test_concentration.py

Single-stock cap (20%) and HHI cap (Σ w_i² ≤ 0.20) must be enforced.
"""
from __future__ import annotations

from simulator.actions import Action, ActionType, QuantityType, ValidationOutcome
from simulator.env import TradingEnvironment


def test_single_stock_cap_enforced(small_market, base_config):
    """A buy that would push a single stock above 20% NAV must be clipped."""
    env = TradingEnvironment(small_market, base_config)
    env.reset()

    # Try to buy 80% of NAV in AAPL (way above 20% cap)
    action = Action(
        action_type=ActionType.BUY,
        ticker="AAPL",
        quantity=0.80,
        quantity_type=QuantityType.NAV_FRACTION,
    )
    obs, state, done, info = env.step([action])

    # After execution, AAPL weight must be <= 20% + small tolerance for fill-price rounding
    nav = state.total_nav
    mv = state.market_value_dict()
    aapl_mv = mv.get("AAPL", 0.0)
    weight = aapl_mv / nav if nav > 0 else 0.0

    assert weight <= base_config.single_stock_cap + 0.02, (
        f"AAPL weight {weight:.2%} exceeds single_stock_cap {base_config.single_stock_cap:.2%}"
    )


def test_hhi_cap_enforced(small_market, base_config):
    """
    After buying into a concentrated position, HHI must not substantially exceed hhi_cap.
    We allow a small execution tolerance because validation uses close price as proxy.
    """
    env = TradingEnvironment(small_market, base_config)
    env.reset()

    # Buy AAPL and MSFT each at 18% of NAV (total 36% — the validator should limit based on HHI)
    actions = [
        Action(action_type=ActionType.BUY, ticker="AAPL",
               quantity=0.18, quantity_type=QuantityType.NAV_FRACTION),
        Action(action_type=ActionType.BUY, ticker="MSFT",
               quantity=0.18, quantity_type=QuantityType.NAV_FRACTION),
    ]
    obs, state, done, info = env.step(actions)

    # HHI should remain close to or below hhi_cap (allow 5% tolerance for price movement)
    assert state.concentration_hhi <= base_config.hhi_cap + 0.05, (
        f"HHI {state.concentration_hhi:.4f} exceeds cap {base_config.hhi_cap}"
    )


def test_sell_always_passes_concentration(small_market, base_config):
    """Selling never violates concentration constraints — sells reduce concentration."""
    env = TradingEnvironment(small_market, base_config)
    env.reset()

    # First buy something
    env.step([Action(action_type=ActionType.BUY, ticker="AAPL",
                     quantity=0.15, quantity_type=QuantityType.NAV_FRACTION)])

    # Then sell it
    obs, state, done, info = env.step([
        Action(action_type=ActionType.SELL, ticker="AAPL",
               quantity_type=QuantityType.CLOSE_ALL)
    ])
    vr = info["validation_results"][0]
    assert vr.outcome != ValidationOutcome.REJECTED, (
        f"SELL was unexpectedly REJECTED: {vr.reason}"
    )
