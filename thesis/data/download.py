"""Download real weekly OHLCV data into MarketReplay's long-format schema.

Price basis
-----------
This downloader exports raw Yahoo Finance weekly ``open/high/low/close/volume``
with ``auto_adjust=False``. That choice keeps the simulator's execution price
basis explicit and avoids silently mixing adjusted closes with raw opens/highs/
lows. Corporate-action handling is therefore *not* normalized away at this
stage; if a study requires split-adjusted execution inputs, that should be
introduced explicitly as a later data-preparation step.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

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

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "sample" / "weekly_ohlcv_real.csv"


def _default_config() -> SimulatorConfig:
    return SimulatorConfig(ticker_universe=list(DEFAULT_TICKERS))


def download_weekly_ohlcv(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download raw weekly OHLCV data with a stable long-format schema."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required. Install it with `pip install yfinance`.") from exc

    frames: list[pd.DataFrame] = []
    failures: list[str] = []

    for raw_ticker in tickers:
        ticker = raw_ticker.strip()
        if not ticker:
            continue

        try:
            frame = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                interval="1wk",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception:
            failures.append(ticker)
            continue

        if frame.empty:
            failures.append(ticker)
            continue

        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [str(column[0]).lower() for column in frame.columns]
        else:
            frame.columns = [str(column).lower() for column in frame.columns]

        required_columns = ["open", "high", "low", "close", "volume"]
        if not set(required_columns).issubset(frame.columns):
            failures.append(ticker)
            continue

        frame = frame.loc[:, required_columns].copy()
        frame = frame.dropna(subset=["open", "high", "low", "close"])
        if frame.empty:
            failures.append(ticker)
            continue

        frame = frame.reset_index().rename(columns={"Date": "date", "date": "date"})
        frame["ticker"] = ticker
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0).astype(int)
        frame = frame.loc[:, ["date", "ticker", "open", "high", "low", "close", "volume"]]
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d")
        frames.append(frame)

    if not frames:
        raise ValueError("No weekly OHLCV data could be downloaded for the requested universe")

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["date", "ticker"]).reset_index(drop=True)
    _validate_downloaded_frame(result)

    if failures:
        print(f"Skipped {len(failures)} tickers with missing/failed downloads: {', '.join(failures)}")

    return result


def save_downloaded_ohlcv(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    config: SimulatorConfig | None = None,
    start: str = "2018-01-01",
    end: str = "2024-12-31",
) -> Path:
    """Download weekly OHLCV data and save it as a single CSV."""
    effective_config = config or _default_config()
    tickers = effective_config.ticker_universe or list(DEFAULT_TICKERS)
    frame = download_weekly_ohlcv(tickers=tickers, start=start, end=end)

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def _validate_downloaded_frame(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("Downloaded OHLCV frame is empty")
    if frame[["open", "high", "low", "close"]].isnull().any().any():
        raise ValueError("Downloaded OHLCV frame contains null prices")
    if (frame[["open", "high", "low", "close"]] <= 0.0).any().any():
        raise ValueError("Downloaded OHLCV frame contains non-positive prices")
    if (frame["volume"] < 0).any():
        raise ValueError("Downloaded OHLCV frame contains negative volume")
    if (frame["high"] < frame["low"]).any():
        raise ValueError("Downloaded OHLCV frame contains rows with high < low")


def main() -> None:
    """CLI entry point for ``python data/download.py``."""
    parser = argparse.ArgumentParser(description="Download weekly OHLCV data via yfinance")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    args = parser.parse_args()

    output_path = save_downloaded_ohlcv(
        output_path=args.output,
        config=_default_config(),
        start=args.start,
        end=args.end,
    )
    print(f"Wrote real weekly OHLCV data to {output_path}")


if __name__ == "__main__":
    main()
