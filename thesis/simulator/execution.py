"""Execution cost estimation and trade fill logic.

Fractional-share policy
-----------------------
This simulator allows fractional shares throughout Stage 2. Share quantities
are therefore represented as floats in action resolution, validation,
execution, and portfolio accounting. This is intentional: it keeps NAV-based
allocation deterministic and avoids introducing a whole-share rounding policy
before that policy is explicitly designed and tested.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from .actions import Action, ActionType, ExecutionResult, QuantityType
from .config import SimulatorConfig
from .state import PortfolioState

if TYPE_CHECKING:
    from .market import MarketReplay


_SHARE_EPSILON = 1e-9


class NonExecutableTradeError(ValueError):
    """Raised when a trade action resolves to an effectively zero execution."""


@dataclass(frozen=True, slots=True)
class ExecutionEstimate:
    """Typed execution-cost estimate used by validation and projected-state logic.

    The object is intentionally lightweight and immutable. It also exposes a
    small mapping-like surface for compatibility with older downstream code.
    """

    signed_shares: float
    reference_price: float
    gross_trade_value: float
    commission_cost: float
    spread_cost: float
    slippage_cost: float
    total_cost: float
    slippage_bps: float
    adv_value: float

    @property
    def trade_value(self) -> float:
        """Backward-compatible alias for legacy callers expecting ``trade_value``."""
        return self.gross_trade_value

    def __getitem__(self, key: str) -> float:
        """Provide dict-style access for compatibility with older code paths."""
        if key == "trade_value":
            return self.trade_value
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Provide a dict-like ``get`` helper for compatibility."""
        try:
            return self[key]
        except AttributeError:
            return default

    def to_dict(self) -> dict[str, float]:
        """Return a plain dictionary representation, including the legacy alias."""
        result = asdict(self)
        result["trade_value"] = self.trade_value
        return result


class ExecutionEngine:
    """Resolve actions into share quantities and deterministic execution outcomes.

    The engine is intentionally side-effect free. It never mutates
    :class:`PortfolioState`; it only converts an action into a signed share
    quantity, estimates execution frictions using data visible at week ``t``,
    and computes the realized fill at ``t + 1`` open.
    """

    def __init__(self, config: SimulatorConfig) -> None:
        self._config = config

    def resolve_shares(
        self,
        action: Action,
        reference_price: float,
        state: PortfolioState,
        batch_start_nav: float,
    ) -> float:
        """Convert an action into a signed share quantity.

        Positive values represent buys. Negative values represent sells.
        ``reference_price`` must be strictly positive and is typically
        ``close[t]`` for estimation or ``open[t+1]`` for actual execution.
        Fractional shares are allowed and preserved.
        """
        if reference_price <= 0.0:
            raise ValueError("reference_price must be > 0")

        ticker = action.ticker or ""
        shares_held = state.shares_dict().get(ticker, 0.0)

        if action.action_type == ActionType.HOLD:
            return 0.0

        if action.action_type == ActionType.BUY:
            assert action.quantity is not None
            assert action.quantity_type is not None
            if action.quantity_type == QuantityType.SHARES:
                return float(action.quantity)
            if action.quantity_type == QuantityType.NOTIONAL_DOLLARS:
                return float(action.quantity) / reference_price
            if action.quantity_type == QuantityType.NAV_FRACTION:
                if batch_start_nav <= 0.0:
                    raise ValueError("batch_start_nav must be > 0 for NAV_FRACTION sizing")
                notional = float(action.quantity) * float(batch_start_nav)
                return notional / reference_price
            raise ValueError(f"Unsupported BUY quantity_type {action.quantity_type!r}")

        if action.action_type == ActionType.SELL:
            assert action.quantity_type is not None
            if action.quantity_type == QuantityType.CLOSE_ALL:
                return -float(shares_held)
            assert action.quantity is not None
            if action.quantity_type == QuantityType.SHARES:
                return -float(action.quantity)
            if action.quantity_type == QuantityType.NOTIONAL_DOLLARS:
                return -(float(action.quantity) / reference_price)
            raise ValueError(f"Unsupported SELL quantity_type {action.quantity_type!r}")

        if action.action_type == ActionType.REDUCE:
            assert action.fraction is not None
            return -(float(shares_held) * float(action.fraction))

        raise ValueError(f"resolve_shares does not support action_type {action.action_type!r}")

    def estimate_cost(
        self,
        action: Action,
        state: PortfolioState,
        t: int,
        market: "MarketReplay",
        batch_start_nav: float,
    ) -> ExecutionEstimate:
        """Estimate gross trade value and execution frictions using ``close[t]``.

        This method is used during validation. It returns a deterministic
        breakdown rather than a single scalar so that validators can reason
        about projected cash, turnover, and clipped action sizes.
        """
        if action.action_type not in (ActionType.BUY, ActionType.SELL, ActionType.REDUCE):
            return self._empty_cost_estimate()

        ticker = action.ticker
        if ticker is None:
            raise ValueError("Trade actions require ticker")

        close_prices = market.get_close_prices(t)
        if ticker not in close_prices:
            raise ValueError(f"No close price available for {ticker!r} at week {t}")

        reference_price = float(close_prices[ticker])
        signed_shares = self.resolve_shares(action, reference_price, state, batch_start_nav)
        return self.estimate_from_signed_shares(
            ticker=ticker,
            signed_shares=signed_shares,
            reference_price=reference_price,
            t=t,
            market=market,
        )

    def estimate_from_signed_shares(
        self,
        ticker: str,
        signed_shares: float,
        reference_price: float,
        t: int,
        market: "MarketReplay",
    ) -> ExecutionEstimate:
        """Estimate execution frictions for a signed share quantity at week ``t``."""
        gross_trade_value = abs(float(signed_shares) * float(reference_price))
        return self._build_cost_estimate(
            ticker=ticker,
            signed_shares=float(signed_shares),
            reference_price=float(reference_price),
            gross_trade_value=gross_trade_value,
            t=t,
            market=market,
        )

    def execute(
        self,
        action: Action,
        t: int,
        market: "MarketReplay",
        state: PortfolioState,
        batch_start_nav: float,
    ) -> ExecutionResult:
        """Execute a validated trade action at ``open[t+1]``."""
        if action.action_type not in (ActionType.BUY, ActionType.SELL, ActionType.REDUCE):
            raise ValueError("execute only supports BUY, SELL, and REDUCE actions")

        execution_week = t + 1
        if execution_week >= market.n_weeks:
            raise IndexError(
                f"Week {execution_week} is outside the available market range for execution"
            )

        ticker = action.ticker
        if ticker is None:
            raise ValueError("Trade actions require ticker")

        open_prices = market.get_open_prices(execution_week)
        if ticker not in open_prices:
            raise ValueError(f"No open price available for {ticker!r} at week {execution_week}")

        execution_price = float(open_prices[ticker])
        executed_shares = self.resolve_shares(action, execution_price, state, batch_start_nav)
        if self.is_effectively_zero_shares(executed_shares):
            raise NonExecutableTradeError(
                "Resolved trade size is effectively zero at execution time"
            )

        estimate = self.estimate_from_signed_shares(
            ticker=ticker,
            signed_shares=executed_shares,
            reference_price=execution_price,
            t=t,
            market=market,
        )
        if self.is_effectively_zero_trade_value(estimate.gross_trade_value):
            raise NonExecutableTradeError(
                "Resolved trade value is effectively zero at execution time"
            )

        return ExecutionResult(
            action=action,
            ticker=ticker,
            executed_shares=executed_shares,
            execution_price=execution_price,
            gross_trade_value=estimate["gross_trade_value"],
            total_cost=estimate["total_cost"],
            commission_cost=estimate["commission_cost"],
            spread_cost=estimate["spread_cost"],
            slippage_cost=estimate["slippage_cost"],
            week_executed=execution_week,
        )

    def _build_cost_estimate(
        self,
        ticker: str,
        signed_shares: float,
        reference_price: float,
        gross_trade_value: float,
        t: int,
        market: "MarketReplay",
    ) -> ExecutionEstimate:
        adv_by_ticker = market.get_adv(t, self._config.adv_lookback_weeks)
        adv_shares = float(adv_by_ticker.get(ticker, 0.0))
        adv_value = adv_shares * float(reference_price)
        if adv_value <= 0.0:
            adv_value = max(gross_trade_value, float(reference_price))

        impact_bps = 0.0
        if gross_trade_value > 0.0:
            impact_bps = (gross_trade_value / adv_value) * self._config.impact_factor * 10_000.0

        slippage_bps = self._config.base_slippage_bps + impact_bps
        commission_cost = gross_trade_value * self._config.commission_rate
        spread_cost = gross_trade_value * self._config.spread_rate
        slippage_cost = gross_trade_value * (slippage_bps / 10_000.0)
        total_cost = commission_cost + spread_cost + slippage_cost

        return ExecutionEstimate(
            signed_shares=float(signed_shares),
            reference_price=float(reference_price),
            gross_trade_value=float(gross_trade_value),
            commission_cost=float(commission_cost),
            spread_cost=float(spread_cost),
            slippage_cost=float(slippage_cost),
            total_cost=float(total_cost),
            slippage_bps=float(slippage_bps),
            adv_value=float(adv_value),
        )

    @staticmethod
    def _empty_cost_estimate() -> ExecutionEstimate:
        return ExecutionEstimate(
            signed_shares=0.0,
            reference_price=0.0,
            gross_trade_value=0.0,
            commission_cost=0.0,
            spread_cost=0.0,
            slippage_cost=0.0,
            total_cost=0.0,
            slippage_bps=0.0,
            adv_value=0.0,
        )

    @staticmethod
    def is_effectively_zero_shares(shares: float) -> bool:
        """Return whether a share quantity is too small to execute."""
        return abs(float(shares)) <= _SHARE_EPSILON

    @staticmethod
    def is_effectively_zero_trade_value(gross_trade_value: float) -> bool:
        """Return whether a trade value is too small to execute."""
        return abs(float(gross_trade_value)) <= _SHARE_EPSILON
