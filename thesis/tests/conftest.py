"""
Shared pytest fixtures for the trading simulator test suite.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is on sys.path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator.config import SimulatorConfig
from simulator.market import MarketReplay


# ------------------------------------------------------------------ #
# Minimal synthetic dataset builder (inline — no file I/O needed)     #
# ------------------------------------------------------------------ #

TICKERS = ["AAPL", "MSFT", "GOOGL"]
N_WEEKS = 20
START_PRICE = {"AAPL": 150.0, "MSFT": 250.0, "GOOGL": 2800.0}


def _build_df(
    tickers: List[str] = TICKERS,
    n_weeks: int = N_WEEKS,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a tiny but valid weekly OHLCV DataFrame for testing."""
    rng = np.random.default_rng(seed)
    rows = []
    from datetime import datetime, timedelta
    start = datetime(2020, 1, 6)
    for i, ticker in enumerate(tickers):
        price = START_PRICE.get(ticker, 100.0 * (i + 1))
        date = start
        for _ in range(n_weeks):
            ret = rng.normal(0.001, 0.03)
            close = max(price * (1 + ret), 1.0)
            open_p = max(price * (1 + rng.normal(0, 0.01)), 1.0)
            high_p = max(open_p, close) * (1 + abs(rng.normal(0, 0.005)))
            low_p  = min(open_p, close) * (1 - abs(rng.normal(0, 0.005)))
            low_p  = max(low_p, 1.0)
            rows.append({
                "date": date,
                "ticker": ticker,
                "open": round(open_p, 4),
                "high": round(high_p, 4),
                "low": round(low_p, 4),
                "close": round(close, 4),
                "volume": int(rng.integers(1_000_000, 5_000_000)),
            })
            price = close
            date = date + timedelta(weeks=1)
    df = pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)
    return df


def _df_to_market(df: pd.DataFrame, universe: List[str]) -> MarketReplay:
    """Write df to an in-memory CSV buffer and load it as a MarketReplay."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    # MarketReplay requires a file path; write to a temp file
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f, index=False)
        tmp_path = f.name

    market = MarketReplay(tmp_path, universe=universe)
    os.unlink(tmp_path)
    return market


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

@pytest.fixture
def small_df():
    return _build_df(tickers=TICKERS, n_weeks=N_WEEKS, seed=0)


@pytest.fixture
def small_market(small_df):
    return _df_to_market(small_df, universe=TICKERS)


@pytest.fixture
def base_config():
    return SimulatorConfig(
        initial_cash=100_000.0,
        ticker_universe=TICKERS,
        max_actions_per_step=5,
        commission_rate=0.001,
        spread_rate=0.0005,
        base_slippage_bps=2.0,
        impact_factor=0.1,
        single_stock_cap=0.20,
        hhi_cap=0.20,
        turnover_cap=0.25,
        vol_budget=0.25,
        vol_lookback_weeks=5,    # short for tests
        observation_history_weeks=52,
        adv_lookback_weeks=2,
        seed=42,
    )
