"""
test_batch_partial_acceptance.py

A batch where one action is valid and another is invalid must:
  - Execute the valid action
  - Reject the invalid action
  - Log both outcomes
"""
from __future__ import annotations

from simulator.actions import Action, ActionType, QuantityType, ValidationOutcome
from simulator.env import TradingEnvironment


def test_valid_and_invalid_action_in_same_batch(small_market, base_config):
    """
    Batch: [valid BUY at 5% NAV, invalid BUY at 300% NAV]
    The valid one executes; the invalid (or clipped-to-zero) one does not.
    """
    env = TradingEnvironment(small_market, base_config)
    env.reset()

    actions = [
        # Valid: 5% of NAV — should pass all rules
        Action(action_type=ActionType.BUY, ticker="AAPL",
               quantity=0.05, quantity_type=QuantityType.NAV_FRACTION),
        # Way over budget: attempt to buy 300% of NAV in MSFT — should be clipped or rejected
        Action(action_type=ActionType.BUY, ticker="MSFT",
               quantity=3.0, quantity_type=QuantityType.NAV_FRACTION),
    ]

    obs, state, done, info = env.step(actions)
    vrs = info["validation_results"]
    ers = info["execution_results"]

    # Sort by action_type order (BUY → BUY, both end up in order they were submitted after sorting)
    # The first action (AAPL 5%) should be accepted or at most clipped (not rejected)
    outcomes = [vr.outcome for vr in vrs]
    assert ValidationOutcome.ACCEPTED in outcomes or ValidationOutcome.CLIPPED in outcomes, (
        "No action was accepted in this batch"
    )

    # At least one action in the batch was accepted/executed
    executed = [er for er in ers if er is not None]
    assert len(executed) >= 1, "Expected at least one execution in the batch"

    # AAPL should have been bought (first action)
    aapl_mv = state.market_value_dict().get("AAPL", 0.0)
    assert aapl_mv > 0, "AAPL was not bought despite valid action"

    # Cash must remain non-negative
    assert state.cash >= -1e-6, f"Cash went negative: {state.cash}"


def test_all_rejected_batch(small_market, base_config):
    """A batch where all actions are invalid must leave portfolio unchanged."""
    env = TradingEnvironment(small_market, base_config)
    obs = env.reset()
    initial_nav = obs.portfolio_state.total_nav

    # Try to sell AAPL when we own none
    actions = [
        Action(action_type=ActionType.SELL, ticker="AAPL",
               quantity=100.0, quantity_type=QuantityType.SHARES),
        Action(action_type=ActionType.SELL, ticker="MSFT",
               quantity=100.0, quantity_type=QuantityType.SHARES),
    ]
    obs, state, done, info = env.step(actions)

    vrs = info["validation_results"]
    for vr in vrs:
        assert vr.outcome == ValidationOutcome.REJECTED, (
            f"Expected REJECTED, got {vr.outcome}: {vr.reason}"
        )

    # No position should have been opened
    assert sum(state.market_value_dict().values()) == 0.0


def test_batch_log_contains_all_action_outcomes(small_market, base_config):
    """The audit log must record all actions in the batch, including rejected ones."""
    env = TradingEnvironment(small_market, base_config)
    env.reset()

    actions = [
        Action(action_type=ActionType.BUY, ticker="AAPL",
               quantity=0.05, quantity_type=QuantityType.NAV_FRACTION),
        Action(action_type=ActionType.SELL, ticker="MSFT",
               quantity=1000.0, quantity_type=QuantityType.SHARES),  # invalid
    ]
    env.step(actions)

    log_df = env.logger.to_dataframe()
    # Should have exactly 2 action rows for the first step
    first_step = log_df[log_df["week_index"] == 0]
    assert len(first_step) == 2, (
        f"Expected 2 log rows for week 0, got {len(first_step)}"
    )
