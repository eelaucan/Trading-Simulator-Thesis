"""Historical market replay with strict week-bounded access."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd


REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


class MarketReplay:
    """Replay interface over a full weekly OHLCV dataset.

    This class is the only component that holds the full market history. Its
    public methods are intentionally week-gated: callers provide a current
    simulation week ``t``, and every accessor returns data drawn only from
    weeks ``<= t``. The private full dataset is never exposed directly.
    """

    def __init__(self, csv_path: str | Path, universe: Sequence[str] | None = None) -> None:
        """Load, validate, and index a long-format weekly OHLCV CSV."""
        path = Path(csv_path).expanduser()
        raw = pd.read_csv(path)
        cleaned = self._prepare_dataframe(raw)
        normalized_universe = (
            self._normalize_ticker_sequence(universe)
            if universe is not None
            else None
        )

        if normalized_universe is not None:
            requested = list(normalized_universe)
            available = set(cleaned["ticker"].unique())
            missing = sorted(set(requested) - available)
            if missing:
                raise ValueError(f"Requested tickers missing from CSV: {missing}")
            cleaned = cleaned[cleaned["ticker"].isin(requested)].copy()

        if cleaned.empty:
            raise ValueError("MarketReplay requires at least one OHLCV row")

        cleaned = cleaned.sort_values(["date", "ticker"]).reset_index(drop=True)

        unique_dates = pd.Index(cleaned["date"].drop_duplicates().sort_values())
        date_to_week_idx = {date_value: index for index, date_value in enumerate(unique_dates)}
        cleaned["_week_idx"] = cleaned["date"].map(date_to_week_idx).astype(int)

        self._data = cleaned
        self._dates: tuple[datetime, ...] = tuple(
            pd.Timestamp(date_value).to_pydatetime() for date_value in unique_dates
        )
        self._available_tickers: tuple[str, ...] = tuple(
            sorted(cleaned["ticker"].drop_duplicates().tolist())
        )

    @property
    def n_weeks(self) -> int:
        """Number of unique simulation weeks available in the dataset."""
        return len(self._dates)

    @property
    def available_tickers(self) -> list[str]:
        """Sorted tickers available in the replay dataset."""
        return list(self._available_tickers)

    @property
    def ticker_universe(self) -> list[str]:
        """Compatibility alias for older code paths."""
        return self.available_tickers

    def get_week_data(self, t: int) -> pd.DataFrame:
        """Return a copy of rows for week ``t`` only."""
        self._validate_week_index(t)
        return self._slice_weeks(t, t)

    def get_history(self, t: int, lookback_weeks: int) -> pd.DataFrame:
        """Return the most recent ``lookback_weeks`` ending at week ``t``.

        The returned frame always satisfies ``_week_idx <= t`` and therefore
        cannot expose future market data. A ``lookback_weeks`` value of ``1``
        returns the current week only.
        """
        self._validate_week_index(t)
        self._validate_positive_window(lookback_weeks, field_name="lookback_weeks")
        start_week = max(0, t - lookback_weeks + 1)
        return self._slice_weeks(start_week, t)

    def get_open_prices(self, t: int) -> dict[str, float]:
        """Return open prices for week ``t`` keyed by ticker."""
        return self._price_map_for_week(t, column="open")

    def get_close_prices(self, t: int) -> dict[str, float]:
        """Return close prices for week ``t`` keyed by ticker."""
        return self._price_map_for_week(t, column="close")

    def get_low_prices(self, t: int) -> dict[str, float]:
        """Return low prices for week ``t`` keyed by ticker."""
        return self._price_map_for_week(t, column="low")

    def get_high_prices(self, t: int) -> dict[str, float]:
        """Return high prices for week ``t`` keyed by ticker."""
        return self._price_map_for_week(t, column="high")

    def get_date(self, t: int) -> datetime:
        """Return the calendar date associated with week ``t``."""
        self._validate_week_index(t)
        return self._dates[t]

    def get_adv(self, t: int, window: int) -> dict[str, float]:
        """Return trailing average volume through week ``t`` by ticker.

        The calculation uses the most recent ``window`` weeks up to and
        including ``t``. No future rows are ever considered.
        """
        history = self.get_history(t=t, lookback_weeks=window)
        grouped = history.groupby("ticker", sort=True)["volume"].mean()
        return {
            ticker: float(grouped.get(ticker, 0.0))
            for ticker in self._available_tickers
        }

    def _prepare_dataframe(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"Missing required market columns: {missing_columns}")

        cleaned = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="raise")
        cleaned["ticker"] = cleaned["ticker"].astype(str).str.strip()

        for column in ("open", "high", "low", "close", "volume"):
            cleaned[column] = pd.to_numeric(cleaned[column], errors="raise")

        if cleaned[list(REQUIRED_COLUMNS)].isnull().any().any():
            raise ValueError("Market data contains nulls in required columns")
        if (cleaned["ticker"] == "").any():
            raise ValueError("Ticker column cannot contain blank values")

        duplicate_mask = cleaned.duplicated(subset=["date", "ticker"])
        if duplicate_mask.any():
            raise ValueError("Duplicate (date, ticker) rows detected in market data")

        if (cleaned[["open", "high", "low", "close"]] <= 0.0).any().any():
            raise ValueError("OHLC prices must be strictly positive")
        if (cleaned["volume"] < 0.0).any():
            raise ValueError("Volume must be non-negative")
        if (cleaned["high"] < cleaned["low"]).any():
            raise ValueError("Each row must satisfy high >= low")
        if ((cleaned["open"] < cleaned["low"]) | (cleaned["open"] > cleaned["high"])).any():
            raise ValueError("Each row must satisfy low <= open <= high")
        if ((cleaned["close"] < cleaned["low"]) | (cleaned["close"] > cleaned["high"])).any():
            raise ValueError("Each row must satisfy low <= close <= high")

        return cleaned

    def _price_map_for_week(self, t: int, column: str) -> dict[str, float]:
        self._validate_week_index(t)
        week_frame = self._slice_weeks(t, t)
        return {
            str(ticker): float(value)
            for ticker, value in zip(week_frame["ticker"], week_frame[column])
        }

    def _slice_weeks(self, start_week: int, end_week: int) -> pd.DataFrame:
        """Return a defensive copy of rows between two inclusive week bounds."""
        if start_week > end_week:
            raise ValueError("start_week cannot be greater than end_week")
        frame = self._data.loc[self._data["_week_idx"].between(start_week, end_week)].copy()
        if frame.empty:
            raise RuntimeError(
                f"No market rows found for inclusive week range [{start_week}, {end_week}]"
            )
        return frame.reset_index(drop=True)

    def _validate_week_index(self, t: int) -> None:
        if not 0 <= t < self.n_weeks:
            raise IndexError(f"Week index {t} is out of range [0, {self.n_weeks - 1}]")

    @staticmethod
    def _validate_positive_window(value: int, field_name: str) -> None:
        if value <= 0:
            raise ValueError(f"{field_name} must be a positive integer")

    @staticmethod
    def _normalize_ticker_sequence(tickers: Sequence[str]) -> tuple[str, ...]:
        if isinstance(tickers, str):
            raise TypeError("Universe tickers must be a sequence of tickers, not a string")
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_ticker in tickers:
            if not isinstance(raw_ticker, str):
                raise TypeError("Universe tickers must be strings")
            ticker = raw_ticker.strip()
            if not ticker:
                raise ValueError("Universe tickers cannot be blank")
            if ticker in seen:
                raise ValueError(f"Universe contains duplicate ticker {ticker!r}")
            seen.add(ticker)
            normalized.append(ticker)
        return tuple(normalized)
