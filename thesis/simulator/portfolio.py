"""Immutable portfolio state transitions and analytics helpers.

Fractional-share policy
-----------------------
Portfolio accounting preserves fractional shares exactly as floats. Stage 2
therefore treats clipped trades, NAV-based buys, and notional conversions as
fractional-share operations unless a later stage explicitly introduces a
whole-share rounding policy.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
import math
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

import numpy as np

from .actions import Action, ActionType, ExecutionResult
from .config import SimulatorConfig
from .observation import PendingLiquidation
from .state import PortfolioState

if TYPE_CHECKING:
    from .market import MarketReplay


_EPSILON = 1e-12


class PortfolioManager:
    """Apply immutable portfolio transitions after validated executions."""

    def __init__(self, config: SimulatorConfig) -> None:
        self._config = config

    def initialize(
        self,
        week_index: int,
        date: datetime,
        ticker_universe: Sequence[str],
    ) -> PortfolioState:
        """Convenience wrapper for constructing the initial all-cash state."""
        return PortfolioState.create_initial(
            initial_cash=self._config.initial_cash,
            start_date=date,
            ticker_universe=ticker_universe,
            week_index=week_index,
        )

    def apply_execution(
        self,
        exec_result: ExecutionResult,
        state: PortfolioState,
        batch_start_nav: Optional[float] = None,
        accumulated_gross_traded: float = 0.0,
    ) -> PortfolioState:
        """Apply one execution result and return a new frozen portfolio state.

        ``batch_start_nav`` is optional so this method can be used standalone in
        smoke checks. When it is provided, ``weekly_turnover`` is updated using
        the fixed batch denominator required by the simulator design. Share
        balances remain fractional floats by design.
        """
        ticker = exec_result.ticker
        shares_delta = float(exec_result.executed_shares)
        execution_price = float(exec_result.execution_price)

        shares = state.shares_dict()
        cost_basis = state.cost_basis_dict()
        market_value = state.market_value_dict()
        unrealized_pnl = state.unrealized_pnl_dict()
        stop_levels = state.stop_levels_dict()
        realized_pnl = float(state.realized_pnl)

        current_shares = float(shares.get(ticker, 0.0))
        current_cost_basis = float(cost_basis.get(ticker, execution_price))

        if shares_delta > 0.0:
            gross_purchase_value = float(exec_result.gross_trade_value)
            new_shares = current_shares + shares_delta
            previous_cost = current_shares * current_cost_basis
            new_cost_basis = (previous_cost + gross_purchase_value) / new_shares
            cash = state.cash - gross_purchase_value - exec_result.total_cost
        else:
            shares_sold = abs(shares_delta)
            if shares_sold > current_shares + _EPSILON:
                raise ValueError(
                    f"Execution sells {shares_sold:.6f} shares of {ticker!r}, "
                    f"but only {current_shares:.6f} are held"
                )
            new_shares = max(0.0, current_shares - shares_sold)
            new_cost_basis = current_cost_basis
            realized_pnl += shares_sold * (execution_price - current_cost_basis)
            cash = state.cash + exec_result.gross_trade_value - exec_result.total_cost

        if new_shares <= _EPSILON:
            shares.pop(ticker, None)
            cost_basis.pop(ticker, None)
            market_value.pop(ticker, None)
            unrealized_pnl.pop(ticker, None)
            stop_levels.pop(ticker, None)
        else:
            shares[ticker] = float(new_shares)
            cost_basis[ticker] = float(new_cost_basis)
            market_value[ticker] = float(new_shares * execution_price)
            unrealized_pnl[ticker] = float(new_shares * (execution_price - new_cost_basis))

        total_nav = float(cash + sum(market_value.values()))
        concentration_hhi = self._compute_hhi_from_market_values(market_value, total_nav)

        weekly_turnover = state.weekly_turnover
        if batch_start_nav is not None:
            if batch_start_nav <= 0.0:
                weekly_turnover = 0.0
            else:
                weekly_turnover = (
                    float(accumulated_gross_traded) + float(exec_result.gross_trade_value)
                ) / float(batch_start_nav)

        return dataclasses.replace(
            state,
            cash=float(cash),
            shares=PortfolioState._to_tuple(shares),
            market_value=PortfolioState._to_tuple(market_value),
            total_nav=total_nav,
            realized_pnl=float(realized_pnl),
            unrealized_pnl=PortfolioState._to_tuple(unrealized_pnl),
            cost_basis=PortfolioState._to_tuple(cost_basis),
            stop_levels=PortfolioState._to_tuple(stop_levels),
            weekly_turnover=float(weekly_turnover),
            concentration_hhi=float(concentration_hhi),
        )

    def mark_to_market(
        self,
        state: PortfolioState,
        t: int,
        date: datetime,
        close_prices: Mapping[str, float],
    ) -> PortfolioState:
        """Revalue the portfolio at week ``t`` close and append NAV history."""
        shares = state.shares_dict()
        cost_basis = state.cost_basis_dict()

        market_value: dict[str, float] = {}
        unrealized_pnl: dict[str, float] = {}
        for ticker, quantity in shares.items():
            if ticker not in close_prices:
                raise ValueError(f"Missing close price for held ticker {ticker!r} at week {t}")
            close_price = float(close_prices[ticker])
            market_value[ticker] = float(quantity * close_price)
            unrealized_pnl[ticker] = float(quantity * (close_price - cost_basis[ticker]))

        total_nav = float(state.cash + sum(market_value.values()))
        nav_history = state.nav_history + (total_nav,)
        concentration_hhi = self._compute_hhi_from_market_values(market_value, total_nav)
        portfolio_volatility = self.compute_rolling_vol(
            nav_history=nav_history,
            vol_lookback_weeks=self._config.vol_lookback_weeks,
        )

        return dataclasses.replace(
            state,
            week_index=int(t),
            date=date,
            market_value=PortfolioState._to_tuple(market_value),
            total_nav=total_nav,
            unrealized_pnl=PortfolioState._to_tuple(unrealized_pnl),
            concentration_hhi=concentration_hhi,
            portfolio_volatility=portfolio_volatility,
            nav_history=nav_history,
        )

    def update_stop(self, action: Action, state: PortfolioState) -> PortfolioState:
        """Apply ``SET_STOP`` or ``REMOVE_STOP`` to the immutable state."""
        ticker = action.ticker
        if ticker is None:
            raise ValueError("Stop actions require ticker")

        stop_levels = state.stop_levels_dict()
        if action.action_type == ActionType.SET_STOP:
            assert action.stop_price is not None
            if state.shares_dict().get(ticker, 0.0) <= _EPSILON:
                raise ValueError(f"Cannot set a stop on {ticker!r} without an active position")
            stop_levels[ticker] = float(action.stop_price)
        elif action.action_type == ActionType.REMOVE_STOP:
            stop_levels.pop(ticker, None)
        else:
            raise ValueError("update_stop only supports SET_STOP and REMOVE_STOP")

        return dataclasses.replace(state, stop_levels=PortfolioState._to_tuple(stop_levels))

    def remove_stop(self, ticker: str, state: PortfolioState) -> PortfolioState:
        """Remove an active stop from a state snapshot if present."""
        stop_levels = state.stop_levels_dict()
        stop_levels.pop(ticker, None)
        return dataclasses.replace(state, stop_levels=PortfolioState._to_tuple(stop_levels))

    def compute_hhi(self, state: PortfolioState) -> float:
        """Compute the concentration HHI from the state's market values."""
        return self._compute_hhi_from_market_values(
            market_values=state.market_value_dict(),
            total_nav=state.total_nav,
        )

    def compute_rolling_vol(
        self,
        nav_history: Sequence[float],
        vol_lookback_weeks: int,
    ) -> Optional[float]:
        """Compute annualized rolling volatility from weekly NAV history."""
        if vol_lookback_weeks <= 0:
            raise ValueError("vol_lookback_weeks must be > 0")
        if len(nav_history) < vol_lookback_weeks + 1:
            return None

        values = np.asarray(nav_history[-(vol_lookback_weeks + 1):], dtype=float)
        if not np.isfinite(values).all() or (values <= 0.0).any():
            return None

        log_returns = np.diff(np.log(values))
        if log_returns.size < 2:
            return None
        return float(np.std(log_returns, ddof=1) * math.sqrt(52.0))

    def check_stop_triggers(
        self,
        state: PortfolioState,
        t: int,
        market: "MarketReplay",
    ) -> list[PendingLiquidation]:
        """Inspect week ``t`` lows and stage stop liquidations for ``t + 1`` open.

        This helper is intended to be called *after* week ``t`` has completed.
        It only reports pending liquidations; it does not mutate state. The
        later environment loop should immediately remove each triggered stop
        from the portfolio state and schedule the returned liquidation for
        execution at ``t + 1`` open. It does not imply same-week execution.
        In the full simulator timeline, calling this helper after week ``t + 1``
        closes will therefore schedule fills for week ``t + 2`` open, which
        matches the project timing rules.
        """
        low_prices = market.get_low_prices(t)
        pending: list[PendingLiquidation] = []
        shares = state.shares_dict()

        for ticker, stop_level in state.stop_levels_dict().items():
            if shares.get(ticker, 0.0) <= _EPSILON:
                continue
            low_price = low_prices.get(ticker)
            if low_price is None:
                continue
            if float(low_price) <= float(stop_level):
                pending.append(
                    PendingLiquidation(
                        ticker=ticker,
                        triggered_by_low=float(low_price),
                        stop_level=float(stop_level),
                        execution_week=t + 1,
                    )
                )

        pending.sort(key=lambda item: item.ticker)
        return pending

    @staticmethod
    def _compute_hhi_from_market_values(
        market_values: Mapping[str, float],
        total_nav: float,
    ) -> float:
        if total_nav <= 0.0:
            return 0.0
        return float(
            sum((value / total_nav) ** 2 for value in market_values.values() if value > 0.0)
        )
