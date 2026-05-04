"""
test_determinism.py

Given identical seed, data, and action sequence, the simulator must produce
identical state paths, logs, and metrics on every run.
"""
from __future__ import annotations

from simulator.actions import Action, ActionType, QuantityType
from simulator.env import TradingEnvironment


def _run_simulation(market, config):
    """Run a fixed action sequence and return (nav_history, log_df)."""
    env = TradingEnvironment(market, config)
    env.reset()

    action_sequence = [
        [Action(action_type=ActionType.BUY, ticker="AAPL",
                quantity=0.10, quantity_type=QuantityType.NAV_FRACTION)],
        [Action(action_type=ActionType.BUY, ticker="MSFT",
                quantity=0.08, quantity_type=QuantityType.NAV_FRACTION)],
        [Action(action_type=ActionType.HOLD)],
        [Action(action_type=ActionType.REDUCE, ticker="AAPL", fraction=0.5)],
        [Action(action_type=ActionType.HOLD)],
    ]

    state = env.current_state
    for i, actions in enumerate(action_sequence):
        if i >= market.n_weeks - 1:
            break
        _, state, done, _ = env.step(actions)
        if done:
            break

    metrics = env.compute_metrics()
    log_df = env.logger.to_dataframe()
    return state.nav_history, log_df, metrics


def test_identical_runs_produce_identical_nav_history(small_market, base_config):
    """Two identical runs must produce byte-for-byte identical NAV histories."""
    nav1, log1, m1 = _run_simulation(small_market, base_config)
    nav2, log2, m2 = _run_simulation(small_market, base_config)

    assert nav1 == nav2, (
        f"NAV histories differ between runs:\n{nav1}\nvs\n{nav2}"
    )


def test_identical_runs_produce_identical_metrics(small_market, base_config):
    """Metrics must be identical across two runs of the same simulation."""
    _, _, m1 = _run_simulation(small_market, base_config)
    _, _, m2 = _run_simulation(small_market, base_config)

    assert m1.total_return == m2.total_return
    assert m1.max_drawdown == m2.max_drawdown
    assert m1.n_clipped_trades == m2.n_clipped_trades
    assert m1.n_invalid_attempts == m2.n_invalid_attempts
    assert m1.n_stop_triggers == m2.n_stop_triggers


def test_identical_runs_produce_identical_logs(small_market, base_config):
    """The audit log (action outcomes, NAV, HHI) must match between runs."""
    _, log1, _ = _run_simulation(small_market, base_config)
    _, log2, _ = _run_simulation(small_market, base_config)

    assert list(log1["total_nav"]) == list(log2["total_nav"]), (
        "total_nav column differs between runs"
    )
    assert list(log1["validation_outcome"]) == list(log2["validation_outcome"]), (
        "validation_outcome column differs between runs"
    )
