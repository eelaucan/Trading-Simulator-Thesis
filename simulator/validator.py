"""Sequential constraint validation for trade and stop actions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Mapping, Optional

import numpy as np

from .actions import Action, ActionType, QuantityType, ValidationOutcome, ValidationResult
from .config import SimulatorConfig
from .state import PortfolioState

if TYPE_CHECKING:
    from .execution import ExecutionEngine, ExecutionEstimate
    from .market import MarketReplay


_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class _ProjectedTradeState:
    """Internal projected post-trade snapshot used during validation."""

    cash: float
    total_nav: float
    gross_trade_value: float
    total_cost: float
    shares: dict[str, float]
    market_values: dict[str, float]
    concentration_hhi: float


class ConstraintValidator:
    """Validate one action against hard simulator rules.

    Validation is sequential rather than batch-atomic. Callers should pass the
    already-projected immutable state from prior accepted/clipped actions in
    the same batch.
    """

    def __init__(self, config: SimulatorConfig, executor: "ExecutionEngine") -> None:
        self._config = config
        self._executor = executor

    def validate(
        self,
        action: Action,
        projected_state: PortfolioState,
        market: "MarketReplay",
        t: int,
        accumulated_turnover_dollars: float,
        batch_start_nav: float,
        estimated_cost: "ExecutionEstimate | Mapping[str, float] | None" = None,
    ) -> ValidationResult:
        """Validate one action and return ACCEPTED, CLIPPED, or REJECTED."""
        vol_rule_active = self._vol_rule_active(projected_state)

        if action.action_type == ActionType.HOLD:
            return ValidationResult(
                original_action=action,
                outcome=ValidationOutcome.ACCEPTED,
                effective_action=action,
                vol_rule_active=vol_rule_active,
            )

        if action.action_type in (ActionType.SET_STOP, ActionType.REMOVE_STOP):
            return self._validate_stop_action(
                action=action,
                projected_state=projected_state,
                market=market,
                t=t,
                vol_rule_active=vol_rule_active,
            )

        ticker = action.ticker
        if ticker is None:
            return self._reject(
                action=action,
                reason="Trade action requires ticker",
                rule_triggered="structural",
                vol_rule_active=vol_rule_active,
            )

        close_prices = market.get_close_prices(t)
        if ticker not in close_prices:
            return self._reject(
                action=action,
                reason=f"No close price available for {ticker!r} at week {t}",
                rule_triggered="no_price",
                vol_rule_active=vol_rule_active,
            )

        base_estimate = self._coerce_estimate(
            estimated_cost
            if estimated_cost is not None
            else self._executor.estimate_cost(
                action=action,
                state=projected_state,
                t=t,
                market=market,
                batch_start_nav=batch_start_nav,
            )
        )
        reference_price = float(base_estimate.reference_price or close_prices[ticker])
        desired_signed_shares = float(base_estimate.signed_shares)

        if self._executor.is_effectively_zero_shares(desired_signed_shares):
            if action.action_type in (ActionType.SELL, ActionType.REDUCE):
                return self._reject(
                    action=action,
                    reason=f"No position in {ticker!r} to sell or reduce",
                    rule_triggered="no_position",
                    vol_rule_active=vol_rule_active,
                )
            return self._reject(
                action=action,
                reason="Resolved trade size is zero",
                rule_triggered="zero_trade",
                vol_rule_active=vol_rule_active,
            )

        current_shares = projected_state.shares_dict().get(ticker, 0.0)
        clipped_signed_shares = desired_signed_shares
        rule_triggered: Optional[str] = None
        reason: Optional[str] = None

        if desired_signed_shares < 0.0:
            if current_shares <= _EPSILON:
                return self._reject(
                    action=action,
                    reason=f"No position in {ticker!r} to sell or reduce",
                    rule_triggered="no_position",
                    vol_rule_active=vol_rule_active,
                )
            max_sellable_shares = float(current_shares)
            if abs(clipped_signed_shares) > max_sellable_shares + _EPSILON:
                clipped_signed_shares = -max_sellable_shares
                rule_triggered = "position_limit"
                reason = f"Clipped to currently held shares in {ticker!r}"

            clipped_signed_shares, turnover_clipped = self._apply_turnover_rule(
                signed_shares=clipped_signed_shares,
                reference_price=reference_price,
                accumulated_turnover_dollars=accumulated_turnover_dollars,
                batch_start_nav=batch_start_nav,
            )
            if turnover_clipped:
                rule_triggered = "turnover_cap"
                reason = "Clipped to remaining weekly turnover budget"
        else:
            clipped_signed_shares, clipped_here = self._clip_by_predicate(
                max_signed_shares=clipped_signed_shares,
                predicate=lambda candidate: self._project_trade(
                    state=projected_state,
                    ticker=ticker,
                    signed_shares=candidate,
                    reference_price=reference_price,
                    t=t,
                    market=market,
                ).cash >= 0.0,
            )
            if clipped_here:
                rule_triggered = "cash_rule"
                reason = "Clipped to available cash after estimated execution costs"

            clipped_signed_shares, clipped_here = self._clip_by_predicate(
                max_signed_shares=clipped_signed_shares,
                predicate=lambda candidate: self._max_stock_weight(
                    self._project_trade(
                        state=projected_state,
                        ticker=ticker,
                        signed_shares=candidate,
                        reference_price=reference_price,
                        t=t,
                        market=market,
                    )
                ) <= self._config.single_stock_cap,
            )
            if clipped_here:
                rule_triggered = "single_stock_cap"
                reason = f"Clipped to respect single-stock cap of {self._config.single_stock_cap:.0%}"

            clipped_signed_shares, clipped_here = self._clip_by_predicate(
                max_signed_shares=clipped_signed_shares,
                predicate=lambda candidate: self._project_trade(
                    state=projected_state,
                    ticker=ticker,
                    signed_shares=candidate,
                    reference_price=reference_price,
                    t=t,
                    market=market,
                ).concentration_hhi <= self._config.hhi_cap,
            )
            if clipped_here:
                rule_triggered = "hhi_cap"
                reason = f"Clipped to respect HHI cap of {self._config.hhi_cap:.3f}"

            clipped_signed_shares, turnover_clipped = self._apply_turnover_rule(
                signed_shares=clipped_signed_shares,
                reference_price=reference_price,
                accumulated_turnover_dollars=accumulated_turnover_dollars,
                batch_start_nav=batch_start_nav,
            )
            if turnover_clipped:
                rule_triggered = "turnover_cap"
                reason = "Clipped to remaining weekly turnover budget"

        if self._executor.is_effectively_zero_shares(clipped_signed_shares):
            return self._reject(
                action=action,
                reason=reason or "Trade size reduced to zero by hard constraints",
                rule_triggered=rule_triggered or "constraint_clip",
                vol_rule_active=vol_rule_active,
            )

        projected_trade = self._project_trade(
            state=projected_state,
            ticker=ticker,
            signed_shares=clipped_signed_shares,
            reference_price=reference_price,
            t=t,
            market=market,
        )

        if projected_trade.total_nav <= 0.0:
            return self._reject(
                action=action,
                reason="Projected total NAV would become non-positive",
                rule_triggered="nav_rule",
                vol_rule_active=vol_rule_active,
            )

        if vol_rule_active:
            projected_volatility = self._projected_portfolio_volatility(
                projected_trade=projected_trade,
                market=market,
                t=t,
                fallback=projected_state.portfolio_volatility,
            )
            if projected_volatility is not None and projected_volatility > self._config.vol_budget:
                return self._reject(
                    action=action,
                    reason=(
                        f"Projected portfolio volatility {projected_volatility:.2%} exceeds "
                        f"budget {self._config.vol_budget:.2%}"
                    ),
                    rule_triggered="vol_budget",
                    vol_rule_active=True,
                )

        effective_action = self._rebuild_action(
            original_action=action,
            signed_shares=clipped_signed_shares,
            reference_price=reference_price,
            state=projected_state,
            batch_start_nav=batch_start_nav,
        )
        clipped = abs(clipped_signed_shares - desired_signed_shares) > _EPSILON

        return ValidationResult(
            original_action=action,
            outcome=ValidationOutcome.CLIPPED if clipped else ValidationOutcome.ACCEPTED,
            effective_action=effective_action,
            reason=reason if clipped else None,
            clip_delta=abs(desired_signed_shares - clipped_signed_shares) if clipped else None,
            vol_rule_active=vol_rule_active,
            rule_triggered=rule_triggered if clipped else None,
        )

    def _validate_stop_action(
        self,
        action: Action,
        projected_state: PortfolioState,
        market: "MarketReplay",
        t: int,
        vol_rule_active: bool,
    ) -> ValidationResult:
        ticker = action.ticker
        assert ticker is not None

        if action.action_type == ActionType.REMOVE_STOP:
            if ticker not in projected_state.stop_levels_dict():
                return self._reject(
                    action=action,
                    reason=f"No active stop exists for {ticker!r}",
                    rule_triggered="no_stop",
                    vol_rule_active=vol_rule_active,
                )
            return ValidationResult(
                original_action=action,
                outcome=ValidationOutcome.ACCEPTED,
                effective_action=action,
                vol_rule_active=vol_rule_active,
            )

        close_prices = market.get_close_prices(t)
        reference_price = close_prices.get(ticker)
        if reference_price is None:
            return self._reject(
                action=action,
                reason=f"No close price available for {ticker!r} at week {t}",
                rule_triggered="no_price",
                vol_rule_active=vol_rule_active,
            )

        if projected_state.shares_dict().get(ticker, 0.0) <= _EPSILON:
            return self._reject(
                action=action,
                reason=f"Cannot set a stop on {ticker!r} without an active position",
                rule_triggered="no_position",
                vol_rule_active=vol_rule_active,
            )

        assert action.stop_price is not None
        lower_bound = reference_price * (1.0 - self._config.stop_max_pct)
        upper_bound = reference_price * (1.0 - self._config.stop_min_pct)
        if not lower_bound <= action.stop_price <= upper_bound:
            return self._reject(
                action=action,
                reason=(
                    f"stop_price {action.stop_price:.4f} must lie within "
                    f"[{lower_bound:.4f}, {upper_bound:.4f}] for {ticker!r}"
                ),
                rule_triggered="stop_band",
                vol_rule_active=vol_rule_active,
            )

        return ValidationResult(
            original_action=action,
            outcome=ValidationOutcome.ACCEPTED,
            effective_action=action,
            vol_rule_active=vol_rule_active,
        )

    def _project_trade(
        self,
        state: PortfolioState,
        ticker: str,
        signed_shares: float,
        reference_price: float,
        t: int,
        market: "MarketReplay",
    ) -> _ProjectedTradeState:
        estimate = self._executor.estimate_from_signed_shares(
            ticker=ticker,
            signed_shares=signed_shares,
            reference_price=reference_price,
            t=t,
            market=market,
        )
        current_shares = state.shares_dict()
        current_market_values = state.market_value_dict()

        shares_after = dict(current_shares)
        market_values_after = dict(current_market_values)

        new_shares = shares_after.get(ticker, 0.0) + signed_shares
        if new_shares <= _EPSILON:
            shares_after.pop(ticker, None)
            market_values_after.pop(ticker, None)
        else:
            shares_after[ticker] = float(new_shares)
            market_values_after[ticker] = float(new_shares * reference_price)

        gross_trade_value = float(estimate.gross_trade_value)
        total_cost = float(estimate.total_cost)
        if signed_shares >= 0.0:
            cash_after = float(state.cash - gross_trade_value - total_cost)
        else:
            cash_after = float(state.cash + gross_trade_value - total_cost)

        total_nav = float(cash_after + sum(market_values_after.values()))
        concentration_hhi = 0.0
        if total_nav > 0.0:
            concentration_hhi = float(
                sum((value / total_nav) ** 2 for value in market_values_after.values() if value > 0.0)
            )

        return _ProjectedTradeState(
            cash=cash_after,
            total_nav=total_nav,
            gross_trade_value=gross_trade_value,
            total_cost=total_cost,
            shares=shares_after,
            market_values=market_values_after,
            concentration_hhi=concentration_hhi,
        )

    @staticmethod
    def _coerce_estimate(
        estimate: "ExecutionEstimate | Mapping[str, float]",
    ) -> "ExecutionEstimate":
        from .execution import ExecutionEstimate

        if isinstance(estimate, ExecutionEstimate):
            return estimate
        return ExecutionEstimate(
            signed_shares=float(estimate.get("signed_shares", 0.0)),
            reference_price=float(estimate.get("reference_price", 0.0)),
            gross_trade_value=float(
                estimate.get("gross_trade_value", estimate.get("trade_value", 0.0))
            ),
            commission_cost=float(estimate.get("commission_cost", estimate.get("commission", 0.0))),
            spread_cost=float(estimate.get("spread_cost", estimate.get("spread", 0.0))),
            slippage_cost=float(estimate.get("slippage_cost", estimate.get("slippage", 0.0))),
            total_cost=float(estimate.get("total_cost", 0.0)),
            slippage_bps=float(estimate.get("slippage_bps", 0.0)),
            adv_value=float(estimate.get("adv_value", 0.0)),
        )

    def _projected_portfolio_volatility(
        self,
        projected_trade: _ProjectedTradeState,
        market: "MarketReplay",
        t: int,
        fallback: Optional[float],
    ) -> Optional[float]:
        if projected_trade.total_nav <= 0.0:
            return fallback

        weights = {
            ticker: value / projected_trade.total_nav
            for ticker, value in projected_trade.market_values.items()
            if value > 0.0
        }
        if not weights:
            return 0.0

        history = market.get_history(t, self._config.vol_lookback_weeks + 1)
        close_history = history.pivot(index="date", columns="ticker", values="close").sort_index()
        required_tickers = [ticker for ticker in weights if ticker in close_history.columns]
        if not required_tickers:
            return fallback

        close_history = close_history.loc[:, required_tickers].dropna()
        if len(close_history) < self._config.vol_lookback_weeks + 1:
            return fallback

        log_returns = np.log(close_history / close_history.shift(1)).dropna()
        if log_returns.empty:
            return fallback

        weight_vector = np.array([weights[ticker] for ticker in required_tickers], dtype=float)
        portfolio_log_returns = log_returns.to_numpy() @ weight_vector
        if portfolio_log_returns.size < 2:
            return fallback

        return float(np.std(portfolio_log_returns, ddof=1) * math.sqrt(52.0))

    def _clip_by_predicate(
        self,
        max_signed_shares: float,
        predicate,
    ) -> tuple[float, bool]:
        if max_signed_shares <= 0.0:
            return max_signed_shares, False
        if predicate(max_signed_shares):
            return max_signed_shares, False
        if not predicate(0.0):
            return 0.0, True

        low = 0.0
        high = max_signed_shares
        for _ in range(60):
            mid = (low + high) / 2.0
            if predicate(mid):
                low = mid
            else:
                high = mid
        return low, True

    def _apply_turnover_rule(
        self,
        signed_shares: float,
        reference_price: float,
        accumulated_turnover_dollars: float,
        batch_start_nav: float,
    ) -> tuple[float, bool]:
        if batch_start_nav <= 0.0:
            return 0.0, abs(signed_shares) > _EPSILON

        turnover_budget = self._config.turnover_cap * batch_start_nav
        remaining_budget = max(0.0, turnover_budget - accumulated_turnover_dollars)
        desired_trade_dollars = abs(signed_shares) * reference_price
        if desired_trade_dollars <= remaining_budget + _EPSILON:
            return signed_shares, False

        if remaining_budget <= 0.0 or reference_price <= 0.0:
            return 0.0, True

        clipped_abs_shares = remaining_budget / reference_price
        return math.copysign(clipped_abs_shares, signed_shares), True

    @staticmethod
    def _max_stock_weight(projected_trade: _ProjectedTradeState) -> float:
        if projected_trade.total_nav <= 0.0:
            return float("inf")
        if not projected_trade.market_values:
            return 0.0
        return max(
            value / projected_trade.total_nav
            for value in projected_trade.market_values.values()
            if value > 0.0
        )

    def _vol_rule_active(self, state: PortfolioState) -> bool:
        """Return whether the rolling-volatility rule is active for this state."""
        return len(state.nav_history) >= self._config.vol_lookback_weeks + 1

    def _rebuild_action(
        self,
        original_action: Action,
        signed_shares: float,
        reference_price: float,
        state: PortfolioState,
        batch_start_nav: float,
    ) -> Action:
        ticker = original_action.ticker
        assert ticker is not None

        if original_action.action_type == ActionType.REDUCE:
            held_shares = state.shares_dict().get(ticker, 0.0)
            if held_shares <= 0.0:
                raise ValueError("Cannot rebuild REDUCE action without current holdings")
            fraction = min(1.0, abs(signed_shares) / held_shares)
            return Action(
                action_type=ActionType.REDUCE,
                ticker=ticker,
                fraction=fraction,
            )

        quantity_type = original_action.quantity_type
        if original_action.action_type == ActionType.BUY:
            assert quantity_type is not None
            if quantity_type == QuantityType.SHARES:
                return Action(
                    action_type=ActionType.BUY,
                    ticker=ticker,
                    quantity=abs(signed_shares),
                    quantity_type=QuantityType.SHARES,
                )
            if quantity_type == QuantityType.NOTIONAL_DOLLARS:
                return Action(
                    action_type=ActionType.BUY,
                    ticker=ticker,
                    quantity=abs(signed_shares) * reference_price,
                    quantity_type=QuantityType.NOTIONAL_DOLLARS,
                )
            if quantity_type == QuantityType.NAV_FRACTION:
                if batch_start_nav <= 0.0:
                    raise ValueError("batch_start_nav must be > 0 for NAV_FRACTION actions")
                return Action(
                    action_type=ActionType.BUY,
                    ticker=ticker,
                    quantity=(abs(signed_shares) * reference_price) / batch_start_nav,
                    quantity_type=QuantityType.NAV_FRACTION,
                )

        if original_action.action_type == ActionType.SELL:
            held_shares = state.shares_dict().get(ticker, 0.0)
            assert quantity_type is not None
            if quantity_type == QuantityType.CLOSE_ALL and abs(signed_shares - (-held_shares)) <= _EPSILON:
                return Action(
                    action_type=ActionType.SELL,
                    ticker=ticker,
                    quantity_type=QuantityType.CLOSE_ALL,
                )
            if quantity_type == QuantityType.NOTIONAL_DOLLARS:
                return Action(
                    action_type=ActionType.SELL,
                    ticker=ticker,
                    quantity=abs(signed_shares) * reference_price,
                    quantity_type=QuantityType.NOTIONAL_DOLLARS,
                )
            return Action(
                action_type=ActionType.SELL,
                ticker=ticker,
                quantity=abs(signed_shares),
                quantity_type=QuantityType.SHARES,
            )

        raise ValueError(f"Unsupported action type {original_action.action_type!r}")

    @staticmethod
    def _reject(
        action: Action,
        reason: str,
        rule_triggered: str,
        vol_rule_active: bool,
    ) -> ValidationResult:
        return ValidationResult(
            original_action=action,
            outcome=ValidationOutcome.REJECTED,
            effective_action=None,
            reason=reason,
            clip_delta=None,
            vol_rule_active=vol_rule_active,
            rule_triggered=rule_triggered,
        )
