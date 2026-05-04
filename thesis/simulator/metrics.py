"""Pure end-of-run analytics for the weekly trading simulator.

The metrics engine is intentionally read-only. It consumes the final frozen
portfolio state and the audit log, then derives summary statistics and
diagnostic tables without mutating simulator components. All performance
metrics are computed from the same realized NAV path that the environment
created through immutable mark-to-market updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np
import pandas as pd

from .config import SimulatorConfig
from .logger import AuditLogger
from .state import PortfolioState


@dataclass(slots=True)
class SimulationMetrics:
    """Final simulation outputs.

    Formulas
    --------
    ``total_return``:
        ``(final_nav / initial_nav) - 1``

    ``max_drawdown``:
        Largest percentage drop from any running NAV peak.

    ``realized_vol``:
        Sample standard deviation of weekly arithmetic returns, annualized by
        ``sqrt(52)``.

    ``sharpe_ratio``:
        Annualized mean weekly excess return divided by weekly return
        volatility, using ``risk_free_rate / 52`` as the weekly risk-free rate.
    """

    total_return: float
    max_drawdown: float
    realized_vol: float
    avg_weekly_turnover: float
    avg_hhi: float
    max_hhi: float
    blow_up_flag: bool
    sharpe_ratio: float | None
    n_invalid_attempts: int
    n_clipped_trades: int
    n_stop_triggers: int
    n_gap_adjustments: int
    vol_rule_activation_week: int | None
    weekly_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    action_log_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation_log_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    execution_log_df: pd.DataFrame = field(default_factory=pd.DataFrame)


class MetricsEngine:
    """Compute final performance, risk, and behaviour metrics."""

    def __init__(self, config: SimulatorConfig) -> None:
        self._config = config

    def compute(
        self,
        final_state: PortfolioState,
        logger: AuditLogger,
    ) -> SimulationMetrics:
        """Compute all configured metrics from the final state and audit log."""
        nav_history = np.asarray(final_state.nav_history, dtype=float)
        initial_nav = (
            float(nav_history[0]) if nav_history.size else float(self._config.initial_cash)
        )
        final_nav = float(final_state.total_nav)
        weekly_returns = self.weekly_returns(nav_history)

        batch_df = logger.to_batch_dataframe()
        action_log_df = logger.to_action_dataframe(include_internal=True)
        validation_log_df = logger.get_validation_log()
        execution_log_df = logger.get_execution_log(include_internal=True)
        stop_trigger_df = logger.to_stop_trigger_dataframe()
        gap_adjustment_df = logger.to_gap_adjustment_dataframe()

        turnover_series = (
            pd.to_numeric(batch_df["weekly_turnover"], errors="coerce")
            if "weekly_turnover" in batch_df.columns
            else pd.Series(dtype=float)
        )
        hhi_series = (
            pd.to_numeric(batch_df["concentration_hhi"], errors="coerce")
            if "concentration_hhi" in batch_df.columns
            else pd.Series(dtype=float)
        )

        return SimulationMetrics(
            total_return=self.total_return(initial_nav=initial_nav, final_nav=final_nav),
            max_drawdown=self.max_drawdown(nav_history),
            realized_vol=self.realized_vol(weekly_returns),
            avg_weekly_turnover=self._nan_safe_mean(turnover_series),
            avg_hhi=self._nan_safe_mean(hhi_series),
            max_hhi=self._nan_safe_max(hhi_series),
            blow_up_flag=self.blow_up_check(nav_history, initial_nav=initial_nav),
            sharpe_ratio=self.sharpe_ratio(weekly_returns),
            n_invalid_attempts=self._count_matches(
                validation_log_df,
                column="validation_outcome",
                expected="rejected",
            ),
            n_clipped_trades=self._count_matches(
                validation_log_df,
                column="validation_outcome",
                expected="clipped",
            ),
            n_stop_triggers=int(len(stop_trigger_df)),
            n_gap_adjustments=int(len(gap_adjustment_df)),
            vol_rule_activation_week=self.vol_rule_activation_week(validation_log_df),
            weekly_returns=weekly_returns,
            action_log_df=action_log_df,
            validation_log_df=validation_log_df,
            execution_log_df=execution_log_df,
        )

    def weekly_returns(self, nav_history: Sequence[float]) -> pd.Series:
        """Return weekly arithmetic returns from the NAV path."""
        nav = np.asarray(nav_history, dtype=float)
        if nav.size < 2:
            return pd.Series(dtype=float)
        returns = (nav[1:] / nav[:-1]) - 1.0
        return pd.Series(returns, index=pd.RangeIndex(start=1, stop=len(nav), step=1), dtype=float)

    @staticmethod
    def total_return(initial_nav: float, final_nav: float) -> float:
        """Compute ``(final_nav / initial_nav) - 1``."""
        if initial_nav <= 0.0:
            return 0.0
        return float((final_nav / initial_nav) - 1.0)

    @staticmethod
    def max_drawdown(nav_history: Sequence[float]) -> float:
        """Compute the maximum drawdown over a NAV path."""
        nav = np.asarray(nav_history, dtype=float)
        if nav.size == 0:
            return 0.0
        running_peak = np.maximum.accumulate(nav)
        drawdowns = 1.0 - np.divide(nav, running_peak, out=np.zeros_like(nav), where=running_peak > 0.0)
        return float(np.max(drawdowns))

    @staticmethod
    def realized_vol(weekly_returns: pd.Series) -> float:
        """Compute annualized realized volatility from weekly returns."""
        if len(weekly_returns) < 2:
            return 0.0
        volatility = weekly_returns.std(ddof=1)
        if pd.isna(volatility):
            return 0.0
        return float(volatility * math.sqrt(52.0))

    def sharpe_ratio(self, weekly_returns: pd.Series) -> float | None:
        """Compute annualized Sharpe ratio from weekly arithmetic returns."""
        if len(weekly_returns) < 2:
            return None
        weekly_rf = float(self._config.risk_free_rate) / 52.0
        excess = weekly_returns - weekly_rf
        volatility = excess.std(ddof=1)
        if pd.isna(volatility) or abs(float(volatility)) < 1e-12:
            return None
        return float(excess.mean() / volatility * math.sqrt(52.0))

    def blow_up_check(
        self,
        nav_history: Sequence[float],
        *,
        initial_nav: float | None = None,
    ) -> bool:
        """Return whether the run breached the configured blow-up thresholds.

        The NAV floor is derived from the realized starting NAV when available
        so the blow-up test is tied to the actual simulated path, not only the
        default config.
        """
        nav = np.asarray(nav_history, dtype=float)
        if nav.size == 0:
            return False
        realized_initial_nav = float(initial_nav) if initial_nav is not None else float(nav[0])
        nav_floor = realized_initial_nav * float(self._config.blow_up_nav_threshold)
        return bool(
            np.any(nav < nav_floor)
            or self.max_drawdown(nav) > float(self._config.blow_up_drawdown_threshold)
        )

    @staticmethod
    def vol_rule_activation_week(validation_log_df: pd.DataFrame) -> int | None:
        """Return the first week where the rolling-vol rule became active."""
        if validation_log_df.empty or "vol_rule_active" not in validation_log_df.columns:
            return None
        active_rows = validation_log_df[validation_log_df["vol_rule_active"] == True]
        if active_rows.empty:
            return None
        return int(active_rows["week_index"].min())

    @staticmethod
    def _count_matches(
        frame: pd.DataFrame,
        *,
        column: str,
        expected: str,
    ) -> int:
        if frame.empty or column not in frame.columns:
            return 0
        return int((frame[column] == expected).sum())

    @staticmethod
    def _nan_safe_mean(series: pd.Series) -> float:
        if series.empty:
            return 0.0
        value = series.mean()
        return 0.0 if pd.isna(value) else float(value)

    @staticmethod
    def _nan_safe_max(series: pd.Series) -> float:
        if series.empty:
            return 0.0
        value = series.max()
        return 0.0 if pd.isna(value) else float(value)
