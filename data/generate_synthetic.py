"""Deterministic synthetic weekly OHLCV data generator."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator.config import SimulatorConfig


DEFAULT_TICKERS: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "JPM",
    "JNJ",
    "XOM",
    "UNH",
    "PG",
    "HD",
    "V",
    "MA",
    "BRK-B",
)

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "sample" / "weekly_ohlcv_synthetic.csv"


def _default_config(seed: int = 42) -> SimulatorConfig:
    return SimulatorConfig(ticker_universe=list(DEFAULT_TICKERS), seed=seed)


def generate_synthetic_ohlcv(
    config: SimulatorConfig,
    n_weeks: int = 104,
    n_tickers: int = 15,
    start_date: str = "2020-01-03",
) -> pd.DataFrame:
    """Generate deterministic long-format weekly OHLCV data.

    The generator is fully deterministic for a given ``SimulatorConfig.seed``.
    Rows are emitted in stable ``date, ticker`` order and validated before
    return so Stage 2 always produces a clean MarketReplay-compatible sample.
    """
    if n_weeks <= 0:
        raise ValueError("n_weeks must be > 0")
    if n_tickers <= 0:
        raise ValueError("n_tickers must be > 0")

    ticker_universe = config.ticker_universe or list(DEFAULT_TICKERS)
    if n_tickers > len(ticker_universe):
        raise ValueError(
            f"Requested {n_tickers} tickers, but only {len(ticker_universe)} are available"
        )
    tickers = ticker_universe[:n_tickers]

    dates = pd.date_range(start=pd.Timestamp(start_date), periods=n_weeks, freq="W-FRI")
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, object]] = []

    for ticker_index, ticker in enumerate(tickers):
        base_price = 25.0 + 12.0 * ticker_index + rng.uniform(0.0, 20.0)
        weekly_drift = 0.001 + 0.0003 * (ticker_index % 5)
        annual_volatility = 0.16 + 0.03 * (ticker_index % 6)
        weekly_volatility = annual_volatility / np.sqrt(52.0)
        base_volume = int(1_500_000 + ticker_index * 275_000 + rng.integers(0, 800_000))

        previous_close = float(base_price)
        for date in dates:
            open_noise = rng.normal(0.0, weekly_volatility * 0.22)
            open_price = max(previous_close * np.exp(open_noise), 0.50)

            close_return = rng.normal(weekly_drift, weekly_volatility)
            close_price = max(open_price * np.exp(close_return), 0.50)

            high_extension = abs(rng.normal(0.0, weekly_volatility * 0.60))
            low_extension = abs(rng.normal(0.0, weekly_volatility * 0.60))
            high_price = max(open_price, close_price) * (1.0 + high_extension)
            low_price = min(open_price, close_price) * max(0.10, 1.0 - low_extension)

            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            low_price = max(low_price, 0.01)

            volume_noise = rng.normal(0.0, 0.18)
            volume = max(int(base_volume * (1.0 + volume_noise)), 1)

            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "open": round(float(open_price), 4),
                    "high": round(float(high_price), 4),
                    "low": round(float(low_price), 4),
                    "close": round(float(close_price), 4),
                    "volume": int(volume),
                }
            )
            previous_close = float(close_price)

    frame = pd.DataFrame(rows, columns=["date", "ticker", "open", "high", "low", "close", "volume"])
    frame = frame.sort_values(["date", "ticker"]).reset_index(drop=True)
    _validate_generated_frame(frame, expected_tickers=tickers, expected_weeks=n_weeks)
    return frame


def save_synthetic_ohlcv(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    config: SimulatorConfig | None = None,
    n_weeks: int = 104,
    n_tickers: int = 15,
    start_date: str = "2020-01-03",
) -> Path:
    """Generate and save deterministic synthetic weekly OHLCV data."""
    effective_config = config or _default_config()
    frame = generate_synthetic_ohlcv(
        config=effective_config,
        n_weeks=n_weeks,
        n_tickers=n_tickers,
        start_date=start_date,
    )
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def _validate_generated_frame(
    frame: pd.DataFrame,
    expected_tickers: list[str],
    expected_weeks: int,
) -> None:
    if frame.empty:
        raise ValueError("Synthetic OHLCV generator produced no rows")
    if frame["ticker"].nunique() != len(expected_tickers):
        raise ValueError("Synthetic OHLCV generator produced an unexpected ticker count")
    if frame["date"].nunique() != expected_weeks:
        raise ValueError("Synthetic OHLCV generator produced an unexpected week count")
    if (frame[["open", "high", "low", "close"]] <= 0.0).any().any():
        raise ValueError("Synthetic OHLCV generator produced non-positive prices")
    if (frame["volume"] <= 0).any():
        raise ValueError("Synthetic OHLCV generator produced non-positive volume")
    if (frame["high"] < frame[["open", "close"]].max(axis=1)).any():
        raise ValueError("Synthetic OHLCV generator produced invalid highs")
    if (frame["low"] > frame[["open", "close"]].min(axis=1)).any():
        raise ValueError("Synthetic OHLCV generator produced invalid lows")


def main() -> None:
    """CLI entry point for ``python data/generate_synthetic.py``."""
    parser = argparse.ArgumentParser(description="Generate deterministic weekly OHLCV sample data")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--weeks", type=int, default=104)
    parser.add_argument("--tickers", type=int, default=15)
    parser.add_argument("--start-date", type=str, default="2020-01-03")
    parser.add_argument("--seed", type=int, default=_default_config().seed)
    args = parser.parse_args()

    config = _default_config(seed=args.seed)

    output_path = save_synthetic_ohlcv(
        output_path=args.output,
        config=config,
        n_weeks=args.weeks,
        n_tickers=args.tickers,
        start_date=args.start_date,
    )
    print(f"Wrote synthetic weekly OHLCV data to {output_path}")


if __name__ == "__main__":
    main()
