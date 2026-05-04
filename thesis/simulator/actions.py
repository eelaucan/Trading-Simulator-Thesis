"""Structured action and result DTOs for the simulator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional


class ActionType(str, Enum):
    """Supported action verbs for the simulator."""

    HOLD = "hold"
    BUY = "buy"
    SELL = "sell"
    REDUCE = "reduce"
    SET_STOP = "set_stop"
    REMOVE_STOP = "remove_stop"


class QuantityType(str, Enum):
    """Supported quantity encodings for trade actions."""

    SHARES = "shares"
    NOTIONAL_DOLLARS = "notional_dollars"
    NAV_FRACTION = "nav_fraction"
    CLOSE_ALL = "close_all"


class ValidationOutcome(str, Enum):
    """Possible outcomes of later-stage action validation."""

    ACCEPTED = "accepted"
    CLIPPED = "clipped"
    REJECTED = "rejected"


@dataclass(slots=True)
class Action:
    """A single structured instruction submitted to the simulator.

    Validation here is intentionally structural only. This class checks that
    the submitted action has the right field shape for its action type, but it
    does not validate market availability, holdings, risk limits, or costs.
    """

    action_type: ActionType
    ticker: Optional[str] = None
    quantity: Optional[float] = None
    quantity_type: Optional[QuantityType] = None
    fraction: Optional[float] = None
    stop_price: Optional[float] = None

    def __post_init__(self) -> None:
        """Enforce structural consistency only, never market or risk rules."""
        self._validate_enum_types()
        self._normalize_ticker()

        if self.action_type == ActionType.HOLD:
            self._require_none("ticker", self.ticker)
            self._require_none("quantity", self.quantity)
            self._require_none("quantity_type", self.quantity_type)
            self._require_none("fraction", self.fraction)
            self._require_none("stop_price", self.stop_price)
            return

        if self.action_type == ActionType.BUY:
            self._require_ticker()
            if self.quantity_type not in {
                QuantityType.SHARES,
                QuantityType.NOTIONAL_DOLLARS,
                QuantityType.NAV_FRACTION,
            }:
                raise ValueError(
                    "BUY requires quantity_type in {SHARES, NOTIONAL_DOLLARS, NAV_FRACTION}"
                )
            self._require_positive_finite_number("quantity", self.quantity)
            self._require_none("fraction", self.fraction)
            self._require_none("stop_price", self.stop_price)
            return

        if self.action_type == ActionType.SELL:
            self._require_ticker()
            if self.quantity_type == QuantityType.CLOSE_ALL:
                self._require_none("quantity", self.quantity)
            else:
                if self.quantity_type not in {
                    QuantityType.SHARES,
                    QuantityType.NOTIONAL_DOLLARS,
                }:
                    raise ValueError(
                        "SELL requires quantity_type in {SHARES, NOTIONAL_DOLLARS} "
                        "or CLOSE_ALL"
                    )
                self._require_positive_finite_number("quantity", self.quantity)
            self._require_none("fraction", self.fraction)
            self._require_none("stop_price", self.stop_price)
            return

        if self.action_type == ActionType.REDUCE:
            self._require_ticker()
            self._require_none("quantity", self.quantity)
            self._require_none("quantity_type", self.quantity_type)
            self._require_none("stop_price", self.stop_price)
            if self.fraction is None or not math.isfinite(float(self.fraction)):
                raise ValueError("REDUCE requires a finite fraction in (0, 1]")
            if not (0.0 < float(self.fraction) <= 1.0):
                raise ValueError("REDUCE requires fraction in (0, 1]")
            return

        if self.action_type == ActionType.SET_STOP:
            self._require_ticker()
            self._require_none("quantity", self.quantity)
            self._require_none("quantity_type", self.quantity_type)
            self._require_none("fraction", self.fraction)
            if self.stop_price is None or not math.isfinite(float(self.stop_price)):
                raise ValueError("SET_STOP requires a finite stop_price > 0")
            if float(self.stop_price) <= 0.0:
                raise ValueError("SET_STOP requires stop_price > 0")
            return

        if self.action_type == ActionType.REMOVE_STOP:
            self._require_ticker()
            self._require_none("quantity", self.quantity)
            self._require_none("quantity_type", self.quantity_type)
            self._require_none("fraction", self.fraction)
            self._require_none("stop_price", self.stop_price)
            return

        raise ValueError(f"Unsupported action_type: {self.action_type}")

    def _require_ticker(self) -> None:
        if self.ticker is None:
            raise ValueError(f"{self.action_type.value} requires ticker")

    def _validate_enum_types(self) -> None:
        if not isinstance(self.action_type, ActionType):
            raise TypeError("action_type must be an ActionType")
        if self.quantity_type is not None and not isinstance(self.quantity_type, QuantityType):
            raise TypeError("quantity_type must be a QuantityType when provided")

    def _normalize_ticker(self) -> None:
        if self.ticker is None:
            return
        if not isinstance(self.ticker, str):
            raise TypeError("ticker must be a string when provided")
        normalized_ticker = self.ticker.strip()
        if not normalized_ticker:
            raise ValueError("ticker must be a non-empty string when provided")
        self.ticker = normalized_ticker

    def _require_positive_finite_number(
        self,
        field_name: str,
        value: Optional[float],
    ) -> None:
        if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(
                f"{self.action_type.value} requires {field_name} to be a finite number > 0"
            )

    @staticmethod
    def _require_none(field_name: str, value: object) -> None:
        if value is not None:
            raise ValueError(f"{field_name} must be omitted for this action type")


@dataclass(slots=True)
class ValidationResult:
    """Outcome of validating one action against simulator rules."""

    original_action: Action
    outcome: ValidationOutcome
    effective_action: Optional[Action]
    reason: Optional[str] = None
    clip_delta: Optional[float] = None
    vol_rule_active: Optional[bool] = None
    rule_triggered: Optional[str] = None


@dataclass(slots=True)
class ExecutionResult:
    """Detailed outcome of an executed trade instruction."""

    action: Action
    ticker: str
    executed_shares: float
    execution_price: float
    gross_trade_value: float
    total_cost: float
    commission_cost: float
    spread_cost: float
    slippage_cost: float
    week_executed: Optional[int] = None
    is_stop_loss: bool = False
    gap_adjusted: bool = False
    original_shares: Optional[float] = None

    @property
    def executed_price(self) -> float:
        """Backward-compatible alias for older execution code."""
        return self.execution_price

    @property
    def trade_value(self) -> float:
        """Backward-compatible alias for older execution code."""
        return self.gross_trade_value

    @property
    def commission(self) -> float:
        """Backward-compatible alias for older execution code."""
        return self.commission_cost

    @property
    def slippage(self) -> float:
        """Backward-compatible alias for older execution code."""
        return self.slippage_cost
