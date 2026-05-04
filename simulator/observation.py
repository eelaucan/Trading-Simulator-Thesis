"""Visible state objects exposed to agents and human users."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from .state import PortfolioState


@dataclass(frozen=True)
class PendingLiquidation:
    """A forced liquidation already triggered by past data and queued for later execution."""

    ticker: str
    triggered_by_low: float
    stop_level: float
    execution_week: int


@dataclass(frozen=True, slots=True)
class Observation:
    """Market and portfolio state visible at week ``week_index``.

    This object must never contain data with ``_week_idx`` greater than the
    current ``week_index``. It is deliberately independent from execution,
    validation, and environment-loop logic.
    """

    week_index: int
    date: datetime
    current_week_ohlcv: pd.DataFrame
    price_history: pd.DataFrame
    portfolio_state: PortfolioState
    available_tickers: list[str]
    pending_liquidations: list[PendingLiquidation] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Copy visible data and assert the observation is free of future rows."""
        current_week_ohlcv = self._copy_dataframe(
            self.current_week_ohlcv,
            field_name="current_week_ohlcv",
        )
        price_history = self._copy_dataframe(
            self.price_history,
            field_name="price_history",
        )
        available_tickers = self._copy_ticker_list(self.available_tickers)
        pending_liquidations = self._copy_pending_liquidations(self.pending_liquidations)

        self._validate_week_frame(
            current_week_ohlcv,
            field_name="current_week_ohlcv",
            exact_week=True,
        )
        self._validate_week_frame(
            price_history,
            field_name="price_history",
            exact_week=False,
        )

        object.__setattr__(self, "current_week_ohlcv", current_week_ohlcv)
        object.__setattr__(self, "price_history", price_history)
        object.__setattr__(self, "available_tickers", available_tickers)
        object.__setattr__(self, "pending_liquidations", pending_liquidations)

    @staticmethod
    def _copy_dataframe(frame: pd.DataFrame, field_name: str) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{field_name} must be a pandas DataFrame")
        return frame.copy(deep=True).reset_index(drop=True)

    @staticmethod
    def _copy_ticker_list(tickers: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_ticker in list(tickers):
            if not isinstance(raw_ticker, str):
                raise TypeError("available_tickers entries must be strings")
            ticker = raw_ticker.strip()
            if not ticker:
                raise ValueError("available_tickers cannot contain blank tickers")
            if ticker in seen:
                raise ValueError(f"available_tickers contains duplicate ticker {ticker!r}")
            seen.add(ticker)
            normalized.append(ticker)
        return normalized

    @staticmethod
    def _copy_pending_liquidations(
        liquidations: list[PendingLiquidation],
    ) -> list[PendingLiquidation]:
        copied = list(liquidations)
        for entry in copied:
            if not isinstance(entry, PendingLiquidation):
                raise TypeError("pending_liquidations must contain PendingLiquidation objects")
        return copied

    def _validate_week_frame(
        self,
        frame: pd.DataFrame,
        field_name: str,
        exact_week: bool,
    ) -> None:
        if "_week_idx" not in frame.columns:
            raise ValueError(f"{field_name} must include an '_week_idx' column")
        if frame.empty:
            raise ValueError(f"{field_name} must not be empty")
        if frame["_week_idx"].isna().any():
            raise ValueError(f"{field_name} contains null values in '_week_idx'")
        if not pd.api.types.is_numeric_dtype(frame["_week_idx"]):
            raise ValueError(f"{field_name} must contain numeric '_week_idx' values")
        if not (frame["_week_idx"] == frame["_week_idx"].astype(int)).all():
            raise ValueError(f"{field_name} must contain integer-like '_week_idx' values")

        week_idx_min = int(frame["_week_idx"].min())
        week_idx_max = int(frame["_week_idx"].max())

        if exact_week and (week_idx_min != self.week_index or week_idx_max != self.week_index):
            raise ValueError("current_week_ohlcv must contain only the current week")
        if not exact_week and week_idx_max != self.week_index:
            raise ValueError("price_history must include the current week and end at week_index")
        if week_idx_min < 0:
            raise ValueError(f"{field_name} cannot contain negative '_week_idx' values")

        if week_idx_max > self.week_index:
            raise ValueError("Observation cannot expose rows from future weeks")
