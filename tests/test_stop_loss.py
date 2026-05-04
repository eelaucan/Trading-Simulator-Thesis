"""
test_stop_loss.py

Stop-loss timeline:
  Week k+1 LOW breaches stop_level
    → PendingLiquidation(execution_week=k+2) created
    → Removed from stop_levels immediately
  Week k+2 step()[1]
    → Position liquidated at k+2 open price
    → Logged with reason='stop_loss_execution'
"""
from __future__ import annotations

import io
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from simulator.config import SimulatorConfig
from simulator.market import MarketReplay
from simulator.env import TradingEnvironment
from simulator.actions import Action, ActionType, QuantityType


def _build_stop_test_market():
    """
    Build a controlled dataset where AAPL crashes sharply in week 3,
    guaranteeing a stop-loss trigger.
    """
    rows = []
    start = datetime(2020, 1, 6)

    def add_week(date, ticker, open_p, high, low, close, volume=1_000_000):
        rows.append({
            "date": date, "ticker": ticker,
            "open": open_p, "high": high, "low": low, "close": close, "volume": volume,
        })

    for w in range(10):
        date = start + timedelta(weeks=w)
        for ticker in ["AAPL", "MSFT"]:
            base = 150.0 if ticker == "AAPL" else 250.0
            if ticker == "AAPL" and w == 3:
                # Sharp drop — LOW drops to 120, well below any 8–20% stop
                add_week(date, ticker, base * 0.95, base * 0.96, 120.0, base * 0.94)
            else:
                add_week(date, ticker, base * 0.999, base * 1.01, base * 0.98, base)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f, index=False)
        tmp = f.name
    market = MarketReplay(tmp, universe=["AAPL", "MSFT"])
    os.unlink(tmp)
    return market


def test_stop_triggered_by_weekly_low():
    """Stop triggers when weekly LOW <= stop_level, not when close <= stop_level."""
    market = _build_stop_test_market()
    cfg = SimulatorConfig(
        initial_cash=100_000.0,
        ticker_universe=["AAPL", "MSFT"],
        vol_lookback_weeks=3,
    )
    env = TradingEnvironment(market, cfg)
    obs = env.reset()

    # Week 0: buy AAPL and set a stop 10% below current close (at ~135)
    aapl_close_0 = market.get_close_prices(0)["AAPL"]
    stop_price = aapl_close_0 * 0.90

    obs, state, done, info = env.step([
        Action(action_type=ActionType.BUY, ticker="AAPL",
               quantity=0.10, quantity_type=QuantityType.NAV_FRACTION),
        Action(action_type=ActionType.SET_STOP, ticker="AAPL", stop_price=stop_price),
    ])

    # Week 1: hold
    obs, state, done, info = env.step([Action(action_type=ActionType.HOLD)])
    assert not done

    # Week 2: hold — this is the step where week 3 data is processed
    # The stop should NOT have triggered yet (week 2 close is fine)
    aapl_shares_before = state.shares_dict().get("AAPL", 0.0)
    obs, state, done, info = env.step([Action(action_type=ActionType.HOLD)])

    # Week 3 LOW (=120) should breach the stop set at ~135
    # PendingLiquidation should have been created
    pending = obs.pending_liquidations
    aapl_pending = [p for p in pending if p.ticker == "AAPL"]

    # Either the stop fired and we have pending liquidations, OR
    # AAPL was already sold (if execution_week has already passed)
    aapl_shares_after = state.shares_dict().get("AAPL", 0.0)
    triggered = len(aapl_pending) > 0 or aapl_shares_after < aapl_shares_before

    assert triggered, (
        f"Stop at {stop_price:.2f} not triggered despite LOW=120. "
        f"AAPL shares before={aapl_shares_before}, after={aapl_shares_after}, "
        f"pending={aapl_pending}"
    )


def test_stop_removed_from_stop_levels_on_trigger():
    """Once a stop triggers, it must be removed from stop_levels to prevent double-trigger."""
    market = _build_stop_test_market()
    cfg = SimulatorConfig(
        initial_cash=100_000.0,
        ticker_universe=["AAPL", "MSFT"],
        vol_lookback_weeks=3,
    )
    env = TradingEnvironment(market, cfg)
    env.reset()

    aapl_close_0 = market.get_close_prices(0)["AAPL"]
    stop_price = aapl_close_0 * 0.90

    env.step([
        Action(action_type=ActionType.BUY, ticker="AAPL",
               quantity=0.10, quantity_type=QuantityType.NAV_FRACTION),
        Action(action_type=ActionType.SET_STOP, ticker="AAPL", stop_price=stop_price),
    ])
    env.step([Action(action_type=ActionType.HOLD)])  # week 1
    obs, state, _, _ = env.step([Action(action_type=ActionType.HOLD)])  # week 2

    # Stop must have been removed from stop_levels
    assert "AAPL" not in state.stop_levels_dict(), (
        "AAPL stop not removed from stop_levels after trigger"
    )
