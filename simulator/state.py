"""Immutable portfolio state for deterministic simulator updates."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime
import math
from typing import Mapping, Optional, Sequence


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """Frozen portfolio snapshot.

    The state is intentionally immutable. Later stages should create updated
    snapshots via :func:`dataclasses.replace` inside a PortfolioManager rather
    than mutating fields in place.
    """

    week_index: int
    date: datetime
    cash: float
    shares: tuple[tuple[str, float], ...]
    market_value: tuple[tuple[str, float], ...]
    total_nav: float
    realized_pnl: float
    unrealized_pnl: tuple[tuple[str, float], ...]
    cost_basis: tuple[tuple[str, float], ...]
    stop_levels: tuple[tuple[str, float], ...]
    weekly_turnover: float
    concentration_hhi: float
    portfolio_volatility: Optional[float]
    nav_history: tuple[float, ...]

    def __post_init__(self) -> None:
        """Canonicalize tuple-backed mappings and validate state shape."""
        if self.week_index < 0:
            raise ValueError("week_index must be >= 0")
        if not isinstance(self.date, datetime):
            raise TypeError("date must be a datetime")

        for field_name in (
            "cash",
            "total_nav",
            "realized_pnl",
            "weekly_turnover",
            "concentration_hhi",
        ):
            self._require_finite(field_name, getattr(self, field_name))

        if self.portfolio_volatility is not None:
            self._require_finite("portfolio_volatility", self.portfolio_volatility)

        for field_name in (
            "shares",
            "market_value",
            "unrealized_pnl",
            "cost_basis",
            "stop_levels",
        ):
            normalized = self._normalize_mapping_pairs(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, normalized)

        normalized_nav_history = tuple(float(value) for value in self.nav_history)
        if not normalized_nav_history:
            raise ValueError("nav_history must contain at least one NAV value")
        for value in normalized_nav_history:
            if not math.isfinite(value):
                raise ValueError("nav_history must contain only finite values")
        object.__setattr__(self, "nav_history", normalized_nav_history)

    @staticmethod
    def _to_tuple(mapping: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
        """Convert a mapping to a deterministic tuple-of-tuples representation."""
        return tuple(sorted((ticker, float(value)) for ticker, value in mapping.items()))

    @classmethod
    def _normalize_mapping_pairs(
        cls,
        pairs: Sequence[tuple[str, float]],
        field_name: str,
    ) -> tuple[tuple[str, float], ...]:
        normalized_pairs: list[tuple[str, float]] = []
        seen_tickers: set[str] = set()
        for raw_ticker, raw_value in pairs:
            if not isinstance(raw_ticker, str):
                raise TypeError(f"{field_name} keys must be strings")
            ticker = raw_ticker.strip()
            if not ticker:
                raise ValueError(f"{field_name} cannot contain blank tickers")
            if ticker in seen_tickers:
                raise ValueError(f"{field_name} contains duplicate ticker {ticker!r}")
            numeric_value = float(raw_value)
            if not math.isfinite(numeric_value):
                raise ValueError(f"{field_name} contains non-finite values")
            seen_tickers.add(ticker)
            normalized_pairs.append((ticker, numeric_value))
        normalized_pairs.sort(key=lambda item: item[0])
        return tuple(normalized_pairs)

    @staticmethod
    def _require_finite(field_name: str, value: float) -> None:
        if not math.isfinite(float(value)):
            raise ValueError(f"{field_name} must be finite")

    @property
    def shares_map(self) -> dict[str, float]:
        """Dictionary view of held shares by ticker."""
        return dict(self.shares)

    @property
    def market_value_map(self) -> dict[str, float]:
        """Dictionary view of marked-to-market position values."""
        return dict(self.market_value)

    @property
    def unrealized_pnl_map(self) -> dict[str, float]:
        """Dictionary view of unrealized PnL by ticker."""
        return dict(self.unrealized_pnl)

    @property
    def cost_basis_map(self) -> dict[str, float]:
        """Dictionary view of average cost basis by ticker."""
        return dict(self.cost_basis)

    @property
    def stop_levels_map(self) -> dict[str, float]:
        """Dictionary view of active stop levels by ticker."""
        return dict(self.stop_levels)

    def shares_dict(self) -> dict[str, float]:
        """Method alias for code that prefers explicit conversion calls."""
        return self.shares_map

    def market_value_dict(self) -> dict[str, float]:
        """Method alias for code that prefers explicit conversion calls."""
        return self.market_value_map

    def unrealized_pnl_dict(self) -> dict[str, float]:
        """Method alias for code that prefers explicit conversion calls."""
        return self.unrealized_pnl_map

    def cost_basis_dict(self) -> dict[str, float]:
        """Method alias for code that prefers explicit conversion calls."""
        return self.cost_basis_map

    def stop_levels_dict(self) -> dict[str, float]:
        """Method alias for code that prefers explicit conversion calls."""
        return self.stop_levels_map

    @classmethod
    def create_initial(
        cls,
        initial_cash: float,
        start_date: datetime,
        ticker_universe: Sequence[str],
        week_index: int = 0,
    ) -> "PortfolioState":
        """Create the initial all-cash state for a simulation run.

        The ticker universe is accepted so initialization is explicit and
        reproducible, even though zero positions are stored sparsely.
        """
        if week_index < 0:
            raise ValueError("week_index must be >= 0")
        if not isinstance(start_date, datetime):
            raise TypeError("start_date must be a datetime")
        if isinstance(ticker_universe, str):
            raise TypeError("ticker_universe must be a sequence of tickers, not a string")
        initial_cash = float(initial_cash)
        if not math.isfinite(initial_cash) or initial_cash < 0.0:
            raise ValueError("initial_cash must be a finite value >= 0")
        seen_tickers: set[str] = set()
        for raw_ticker in ticker_universe:
            if not isinstance(raw_ticker, str) or not raw_ticker.strip():
                raise ValueError("ticker_universe must contain only non-empty strings")
            ticker = raw_ticker.strip()
            if ticker in seen_tickers:
                raise ValueError(f"ticker_universe contains duplicate ticker {ticker!r}")
            seen_tickers.add(ticker)
        empty_mapping: tuple[tuple[str, float], ...] = tuple()
        return cls(
            week_index=week_index,
            date=start_date,
            cash=initial_cash,
            shares=empty_mapping,
            market_value=empty_mapping,
            total_nav=initial_cash,
            realized_pnl=0.0,
            unrealized_pnl=empty_mapping,
            cost_basis=empty_mapping,
            stop_levels=empty_mapping,
            weekly_turnover=0.0,
            concentration_hhi=0.0,
            portfolio_volatility=None,
            nav_history=(initial_cash,),
        )

    @classmethod
    def initial(
        cls,
        week_index: int,
        date: datetime,
        initial_cash: float,
        ticker_universe: Sequence[str],
    ) -> "PortfolioState":
        """Compatibility wrapper around :meth:`create_initial`."""
        return cls.create_initial(
            initial_cash=initial_cash,
            start_date=date,
            ticker_universe=ticker_universe,
            week_index=week_index,
        )

    def replace(self, **kwargs: object) -> "PortfolioState":
        """Return a new state with selected fields updated."""
        return dataclasses.replace(self, **kwargs)
