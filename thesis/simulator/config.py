"""Central simulator configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(slots=True)
class SimulatorConfig:
    """Single source of truth for simulator parameters and thresholds.

    ``initial_decision_week`` controls when the participant first sees a
    decision screen. When it is ``None``, the environment chooses a sensible
    default from ``observation_history_weeks`` so the first decision can be
    made with historical context already visible.
    """

    initial_cash: float = 100_000.0
    ticker_universe: list[str] = field(default_factory=list)
    max_actions_per_step: int = 5

    commission_rate: float = 0.001
    spread_rate: float = 0.0005
    base_slippage_bps: float = 2.0
    impact_factor: float = 0.1

    single_stock_cap: float = 0.20
    hhi_cap: float = 0.20
    turnover_cap: float = 0.25
    vol_budget: float = 0.25
    vol_lookback_weeks: int = 12
    stop_min_pct: float = 0.08
    stop_max_pct: float = 0.20
    cash_buffer: float = 0.0

    observation_history_weeks: int = 52
    adv_lookback_weeks: int = 4
    initial_decision_week: int | None = None

    blow_up_nav_threshold: float = 0.70
    blow_up_drawdown_threshold: float = 0.40

    risk_free_rate: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate basic parameter sanity for a deterministic simulator run."""
        self.ticker_universe = self._normalize_ticker_universe(self.ticker_universe)

        self._require_finite("initial_cash", self.initial_cash)
        self._require_finite("risk_free_rate", self.risk_free_rate)
        self._require_integer("max_actions_per_step", self.max_actions_per_step, minimum=1)
        self._require_integer("vol_lookback_weeks", self.vol_lookback_weeks, minimum=1)
        self._require_integer(
            "observation_history_weeks",
            self.observation_history_weeks,
            minimum=1,
        )
        self._require_integer("adv_lookback_weeks", self.adv_lookback_weeks, minimum=1)
        if self.initial_decision_week is not None:
            self._require_integer(
                "initial_decision_week",
                self.initial_decision_week,
                minimum=0,
            )
        self._require_integer("seed", self.seed)

        for field_name in (
            "commission_rate",
            "spread_rate",
            "base_slippage_bps",
            "impact_factor",
            "cash_buffer",
        ):
            self._require_finite_non_negative(field_name, getattr(self, field_name))

        for field_name in (
            "single_stock_cap",
            "hhi_cap",
            "turnover_cap",
            "vol_budget",
            "stop_min_pct",
            "stop_max_pct",
            "blow_up_nav_threshold",
            "blow_up_drawdown_threshold",
        ):
            self._require_fraction(field_name, getattr(self, field_name))

        if self.stop_min_pct > self.stop_max_pct:
            raise ValueError("stop_min_pct must be less than or equal to stop_max_pct")

    @staticmethod
    def _normalize_ticker_universe(tickers: list[str]) -> list[str]:
        if isinstance(tickers, str):
            raise TypeError("ticker_universe must be a sequence of tickers, not a string")
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_ticker in list(tickers):
            if not isinstance(raw_ticker, str):
                raise TypeError("ticker_universe entries must be strings")
            ticker = raw_ticker.strip()
            if not ticker:
                raise ValueError("ticker_universe cannot contain blank tickers")
            if ticker in seen:
                raise ValueError(f"ticker_universe contains duplicate ticker {ticker!r}")
            seen.add(ticker)
            normalized.append(ticker)
        return normalized

    @staticmethod
    def _require_integer(field_name: str, value: int, minimum: int | None = None) -> None:
        if not isinstance(value, int):
            raise TypeError(f"{field_name} must be an integer")
        if minimum is not None and value < minimum:
            raise ValueError(f"{field_name} must be >= {minimum}")

    @staticmethod
    def _require_finite(field_name: str, value: float) -> None:
        if not math.isfinite(float(value)):
            raise ValueError(f"{field_name} must be finite")

    @classmethod
    def _require_finite_non_negative(cls, field_name: str, value: float) -> None:
        cls._require_finite(field_name, value)
        if float(value) < 0.0:
            raise ValueError(f"{field_name} must be >= 0")

    @classmethod
    def _require_fraction(cls, field_name: str, value: float) -> None:
        cls._require_finite(field_name, value)
        numeric_value = float(value)
        if not 0.0 <= numeric_value <= 1.0:
            raise ValueError(f"{field_name} must be between 0 and 1 inclusive")
