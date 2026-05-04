"""Deterministic autonomous benchmark agent for weekly simulator episodes.

The agent is intentionally simple: it reads the current observation, computes
trailing momentum and volatility signals, maps those signals into target
weights, and emits structured simulator actions. No external data, model API,
browser automation, reinforcement learning, or fitting step is used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from simulator.actions import Action, ActionType, QuantityType
from simulator.observation import Observation


_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class BenchmarkAgentConfig:
    """Transparent parameters for the rule-based benchmark agent."""

    short_momentum_window: int = 4
    medium_momentum_window: int = 12
    volatility_window: int = 12
    short_momentum_weight: float = 0.6
    medium_momentum_weight: float = 0.4
    volatility_penalty: float = 0.5
    min_score: float = 0.0
    cash_reserve: float = 0.05
    max_position_weight: float = 0.30
    max_positions: int = 3
    rebalance_threshold: float = 0.03
    max_turnover: float = 0.25
    max_actions_per_step: int = 5

    def __post_init__(self) -> None:
        """Validate parameters once so later decisions stay small and direct."""
        for field_name in (
            "short_momentum_window",
            "medium_momentum_window",
            "volatility_window",
            "max_positions",
            "max_actions_per_step",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")

        for field_name in (
            "short_momentum_weight",
            "medium_momentum_weight",
            "volatility_penalty",
            "min_score",
            "cash_reserve",
            "max_position_weight",
            "rebalance_threshold",
            "max_turnover",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")

        if self.short_momentum_window > self.medium_momentum_window:
            raise ValueError("short_momentum_window must be <= medium_momentum_window")
        if not 0.0 <= self.cash_reserve < 1.0:
            raise ValueError("cash_reserve must be in [0, 1)")
        if not 0.0 < self.max_position_weight <= 1.0:
            raise ValueError("max_position_weight must be in (0, 1]")
        if not 0.0 <= self.rebalance_threshold <= 1.0:
            raise ValueError("rebalance_threshold must be in [0, 1]")
        if not 0.0 < self.max_turnover <= 1.0:
            raise ValueError("max_turnover must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class TickerSignal:
    """Decision-time signal values for one ticker."""

    ticker: str
    weeks_observed: int
    latest_close: float | None
    momentum_4w: float | None
    momentum_12w: float | None
    volatility: float | None
    score: float | None
    eligible: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly signal record."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _DecisionRecord:
    """Structured weekly agent decision saved for thesis analysis."""

    week_index: int
    date: datetime
    candidate_scores: tuple[dict[str, Any], ...]
    selected_tickers: tuple[str, ...]
    holdings_before: dict[str, dict[str, float]]
    target_weights: dict[str, float]
    estimated_post_decision_weights: dict[str, float]
    generated_actions: tuple[dict[str, Any], ...]
    reasons_by_ticker: dict[str, tuple[str, ...]]
    remaining_cash_weight_estimate: float
    remaining_turnover_budget: float

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""
        return asdict(self)


class AutonomousBenchmarkAgent:
    """Weekly long-only rule-based benchmark agent.

    Public use is deliberately small:

    ``decide(observation) -> list[Action]``

    Each call also appends one structured decision record that can be exported
    as CSV or JSONL after the episode.
    """

    def __init__(self, config: BenchmarkAgentConfig | None = None) -> None:
        self.config = config or BenchmarkAgentConfig()
        self._decision_records: list[_DecisionRecord] = []

    @property
    def decision_records(self) -> tuple[dict[str, Any], ...]:
        """Immutable JSON-friendly view of logged weekly decisions."""
        return tuple(record.to_dict() for record in self._decision_records)

    def reset_log(self) -> None:
        """Clear accumulated decision records before a fresh episode."""
        self._decision_records.clear()

    def compute_signals(self, observation: Observation) -> list[TickerSignal]:
        """Compute trailing score components from the current observation only."""
        tickers = list(getattr(observation, "available_tickers", ()))
        history = getattr(observation, "price_history", pd.DataFrame())
        week_index = int(getattr(observation, "week_index", 0))

        if not isinstance(history, pd.DataFrame) or history.empty:
            return [
                TickerSignal(
                    ticker=ticker,
                    weeks_observed=0,
                    latest_close=None,
                    momentum_4w=None,
                    momentum_12w=None,
                    volatility=None,
                    score=None,
                    eligible=False,
                    reason="insufficient_history",
                )
                for ticker in tickers
            ]

        required_columns = {"ticker", "close", "_week_idx"}
        if not required_columns.issubset(history.columns):
            return [
                TickerSignal(
                    ticker=ticker,
                    weeks_observed=0,
                    latest_close=None,
                    momentum_4w=None,
                    momentum_12w=None,
                    volatility=None,
                    score=None,
                    eligible=False,
                    reason="missing_history_columns",
                )
                for ticker in tickers
            ]

        visible_history = history.loc[history["_week_idx"] <= week_index].copy()
        visible_history = visible_history[visible_history["ticker"].isin(tickers)]
        visible_history = visible_history.sort_values(["ticker", "_week_idx"])

        return [self._signal_for_ticker(ticker, visible_history) for ticker in tickers]

    def decide(self, observation: Observation) -> list[Action]:
        """Return one weekly batch of valid simulator actions."""
        signals = self.compute_signals(observation)
        ranked_positive = self._rank_positive_signals(signals)
        selected_tickers = tuple(signal.ticker for signal in ranked_positive[: self.config.max_positions])

        state = observation.portfolio_state
        nav = max(float(state.total_nav), _EPSILON)
        current_weights = self._current_weights(observation)
        projected_weights = dict(current_weights)
        holdings_before = self._holdings_before(observation)
        target_weights = self._target_weights(selected_tickers)
        reasons_by_ticker: dict[str, list[str]] = {
            ticker: [] for ticker in observation.available_tickers
        }

        remaining_turnover = float(self.config.max_turnover)
        projected_cash_weight = max(0.0, float(state.cash) / nav)
        actions: list[Action] = []
        action_reasons: list[tuple[Action, tuple[str, ...]]] = []

        signal_by_ticker = {signal.ticker: signal for signal in signals}
        held_tickers = [
            ticker
            for ticker, shares in sorted(state.shares_dict().items())
            if shares > _EPSILON and ticker in observation.available_tickers
        ]

        for ticker in held_tickers:
            if ticker in selected_tickers:
                continue
            reason = self._exit_reason(signal_by_ticker.get(ticker))
            sell_action, sold_weight, sell_reasons = self._build_sell_action(
                observation=observation,
                ticker=ticker,
                desired_sell_weight=current_weights.get(ticker, 0.0),
                remaining_turnover=remaining_turnover,
                primary_reason=reason,
            )
            reasons_by_ticker.setdefault(ticker, []).extend(sell_reasons)
            if sell_action is None:
                continue
            actions.append(sell_action)
            action_reasons.append((sell_action, tuple(sell_reasons)))
            remaining_turnover = max(0.0, remaining_turnover - sold_weight)
            projected_cash_weight += sold_weight
            projected_weights[ticker] = max(0.0, projected_weights.get(ticker, 0.0) - sold_weight)
            if len(actions) >= self.config.max_actions_per_step:
                self._mark_action_limit(reasons_by_ticker, observation.available_tickers)
                break

        if len(actions) < self.config.max_actions_per_step:
            for ticker in selected_tickers:
                target_weight = target_weights[ticker]
                current_weight = projected_weights.get(ticker, 0.0)
                above_hard_agent_cap = current_weight > self.config.max_position_weight + _EPSILON
                above_rebalance_band = current_weight > target_weight + self.config.rebalance_threshold
                if not above_hard_agent_cap and not above_rebalance_band:
                    continue

                desired_sell_weight = current_weight - target_weight
                sell_action, sold_weight, sell_reasons = self._build_sell_action(
                    observation=observation,
                    ticker=ticker,
                    desired_sell_weight=desired_sell_weight,
                    remaining_turnover=remaining_turnover,
                    primary_reason="concentration_limit",
                )
                reasons_by_ticker.setdefault(ticker, []).extend(sell_reasons)
                if sell_action is None:
                    continue
                actions.append(sell_action)
                action_reasons.append((sell_action, tuple(sell_reasons)))
                remaining_turnover = max(0.0, remaining_turnover - sold_weight)
                projected_cash_weight += sold_weight
                projected_weights[ticker] = max(0.0, current_weight - sold_weight)
                if len(actions) >= self.config.max_actions_per_step:
                    self._mark_action_limit(reasons_by_ticker, observation.available_tickers)
                    break

        if len(actions) < self.config.max_actions_per_step:
            for signal in ranked_positive:
                ticker = signal.ticker
                target_weight = target_weights.get(ticker, 0.0)
                current_weight = projected_weights.get(ticker, 0.0)
                if target_weight <= current_weight + self.config.rebalance_threshold:
                    reasons_by_ticker.setdefault(ticker, []).append("rebalance_threshold")
                    continue

                available_cash_weight = projected_cash_weight - self.config.cash_reserve
                desired_buy_weight = target_weight - current_weight
                buy_weight = min(desired_buy_weight, available_cash_weight, remaining_turnover)
                if buy_weight <= _EPSILON:
                    if available_cash_weight <= _EPSILON:
                        reasons_by_ticker.setdefault(ticker, []).append("insufficient_cash")
                    if remaining_turnover <= _EPSILON:
                        reasons_by_ticker.setdefault(ticker, []).append("turnover_limit")
                    continue

                buy_reasons = ["positive_momentum"]
                if buy_weight < desired_buy_weight - _EPSILON:
                    if available_cash_weight <= buy_weight + _EPSILON:
                        buy_reasons.append("insufficient_cash")
                    if remaining_turnover <= buy_weight + _EPSILON:
                        buy_reasons.append("turnover_limit")

                action = Action(
                    action_type=ActionType.BUY,
                    ticker=ticker,
                    quantity=float(buy_weight),
                    quantity_type=QuantityType.NAV_FRACTION,
                )
                actions.append(action)
                action_reasons.append((action, tuple(buy_reasons)))
                reasons_by_ticker.setdefault(ticker, []).extend(buy_reasons)
                projected_cash_weight = max(0.0, projected_cash_weight - buy_weight)
                remaining_turnover = max(0.0, remaining_turnover - buy_weight)
                projected_weights[ticker] = current_weight + buy_weight
                if len(actions) >= self.config.max_actions_per_step:
                    self._mark_action_limit(reasons_by_ticker, observation.available_tickers)
                    break

        if not actions:
            hold_reason = (
                "hold_no_qualified_asset"
                if not selected_tickers
                else "rebalance_threshold"
            )
            actions = [Action(action_type=ActionType.HOLD)]
            action_reasons = [(actions[0], (hold_reason,))]
            for ticker in observation.available_tickers:
                reasons_by_ticker.setdefault(ticker, []).append(hold_reason)

        self._log_decision(
            observation=observation,
            signals=signals,
            selected_tickers=selected_tickers,
            holdings_before=holdings_before,
            target_weights=target_weights,
            estimated_post_decision_weights=projected_weights,
            actions_and_reasons=action_reasons,
            reasons_by_ticker={
                ticker: tuple(dict.fromkeys(reason for reason in reasons if reason))
                for ticker, reasons in reasons_by_ticker.items()
            },
            remaining_cash_weight_estimate=projected_cash_weight,
            remaining_turnover_budget=remaining_turnover,
        )
        return actions

    def to_decision_dataframe(self) -> pd.DataFrame:
        """Return one CSV-friendly row per candidate ticker per decision week."""
        rows: list[dict[str, Any]] = []
        for record in self._decision_records:
            generated_actions_json = json.dumps(
                [self._json_ready(action) for action in record.generated_actions],
                sort_keys=True,
            )
            selected = set(record.selected_tickers)
            holdings_json = json.dumps(self._json_ready(record.holdings_before), sort_keys=True)
            action_by_ticker = {
                str(action.get("ticker")): action
                for action in record.generated_actions
                if action.get("ticker") is not None
            }
            hold_actions = [
                action for action in record.generated_actions
                if action.get("action_type") == ActionType.HOLD.value
            ]

            for candidate in record.candidate_scores:
                ticker = str(candidate["ticker"])
                action_payload = action_by_ticker.get(ticker)
                if action_payload is None and hold_actions:
                    action_payload = hold_actions[0]
                rows.append(
                    {
                        "week_index": record.week_index,
                        "date": record.date,
                        "ticker": ticker,
                        "weeks_observed": candidate["weeks_observed"],
                        "latest_close": candidate["latest_close"],
                        "momentum_4w": candidate["momentum_4w"],
                        "momentum_12w": candidate["momentum_12w"],
                        "volatility": candidate["volatility"],
                        "score": candidate["score"],
                        "eligible": candidate["eligible"],
                        "signal_reason": candidate["reason"],
                        "selected": ticker in selected,
                        "current_weight": record.holdings_before.get(ticker, {}).get("weight", 0.0),
                        "target_weight": record.target_weights.get(ticker, 0.0),
                        "estimated_post_decision_weight": (
                            record.estimated_post_decision_weights.get(ticker, 0.0)
                        ),
                        "reasons": json.dumps(record.reasons_by_ticker.get(ticker, ())),
                        "action_type": (
                            None if action_payload is None else action_payload.get("action_type")
                        ),
                        "action_quantity": (
                            None if action_payload is None else action_payload.get("quantity")
                        ),
                        "action_quantity_type": (
                            None if action_payload is None else action_payload.get("quantity_type")
                        ),
                        "generated_actions": generated_actions_json,
                        "holdings_before": holdings_json,
                    }
                )
        return pd.DataFrame(rows)

    def export_decision_csv(self, path: str | Path) -> None:
        """Export the flattened decision log as CSV."""
        output_path = Path(path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_decision_dataframe().to_csv(output_path, index=False)

    def export_decision_jsonl(self, path: str | Path) -> None:
        """Export one nested decision record per line as JSONL."""
        output_path = Path(path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in self._decision_records:
                handle.write(json.dumps(self._json_ready(record.to_dict()), sort_keys=True))
                handle.write("\n")

    def _signal_for_ticker(self, ticker: str, history: pd.DataFrame) -> TickerSignal:
        ticker_history = history[history["ticker"] == ticker].sort_values("_week_idx")
        closes = pd.to_numeric(ticker_history["close"], errors="coerce").dropna().astype(float)
        closes = closes[closes > 0.0].reset_index(drop=True)
        weeks_observed = int(len(closes))
        latest_close = float(closes.iloc[-1]) if weeks_observed else None

        required_weeks = self.config.medium_momentum_window + 1
        if weeks_observed < required_weeks:
            return TickerSignal(
                ticker=ticker,
                weeks_observed=weeks_observed,
                latest_close=latest_close,
                momentum_4w=None,
                momentum_12w=None,
                volatility=None,
                score=None,
                eligible=False,
                reason="insufficient_history",
            )

        short_momentum = self._window_return(closes, self.config.short_momentum_window)
        medium_momentum = self._window_return(closes, self.config.medium_momentum_window)
        returns = closes.pct_change().dropna().tail(self.config.volatility_window)
        volatility = float(returns.std(ddof=1)) if len(returns) >= 2 else 0.0
        score = (
            self.config.short_momentum_weight * short_momentum
            + self.config.medium_momentum_weight * medium_momentum
            - self.config.volatility_penalty * volatility
        )
        eligible = bool(score > self.config.min_score)
        return TickerSignal(
            ticker=ticker,
            weeks_observed=weeks_observed,
            latest_close=latest_close,
            momentum_4w=float(short_momentum),
            momentum_12w=float(medium_momentum),
            volatility=float(volatility),
            score=float(score),
            eligible=eligible,
            reason="positive_momentum" if eligible else "non_positive_score",
        )

    @staticmethod
    def _window_return(closes: pd.Series, window: int) -> float:
        current = float(closes.iloc[-1])
        previous = float(closes.iloc[-(window + 1)])
        if previous <= 0.0:
            return 0.0
        return float((current / previous) - 1.0)

    def _rank_positive_signals(self, signals: Sequence[TickerSignal]) -> list[TickerSignal]:
        positive = [
            signal for signal in signals
            if signal.eligible and signal.score is not None and signal.score > self.config.min_score
        ]
        return sorted(positive, key=lambda signal: (-float(signal.score or 0.0), signal.ticker))

    def _target_weights(self, selected_tickers: Sequence[str]) -> dict[str, float]:
        if not selected_tickers:
            return {}
        investable_weight = max(0.0, 1.0 - self.config.cash_reserve)
        equal_weight = investable_weight / float(len(selected_tickers))
        per_position = min(float(self.config.max_position_weight), equal_weight)
        return {ticker: float(per_position) for ticker in selected_tickers}

    @staticmethod
    def _current_weights(observation: Observation) -> dict[str, float]:
        state = observation.portfolio_state
        nav = max(float(state.total_nav), _EPSILON)
        weights = {ticker: 0.0 for ticker in observation.available_tickers}
        for ticker, value in state.market_value_dict().items():
            if ticker in weights:
                weights[ticker] = max(0.0, float(value) / nav)
        return weights

    @staticmethod
    def _holdings_before(observation: Observation) -> dict[str, dict[str, float]]:
        state = observation.portfolio_state
        nav = max(float(state.total_nav), _EPSILON)
        shares = state.shares_dict()
        values = state.market_value_dict()
        return {
            ticker: {
                "shares": float(shares[ticker]),
                "market_value": float(values.get(ticker, 0.0)),
                "weight": float(values.get(ticker, 0.0)) / nav,
            }
            for ticker in sorted(shares)
            if shares[ticker] > _EPSILON and ticker in observation.available_tickers
        }

    @staticmethod
    def _exit_reason(signal: TickerSignal | None) -> str:
        if signal is None or not signal.eligible:
            return "negative_signal_exit"
        return "rank_exit"

    def _build_sell_action(
        self,
        *,
        observation: Observation,
        ticker: str,
        desired_sell_weight: float,
        remaining_turnover: float,
        primary_reason: str,
    ) -> tuple[Action | None, float, list[str]]:
        state = observation.portfolio_state
        shares_held = float(state.shares_dict().get(ticker, 0.0))
        current_weight = self._current_weights(observation).get(ticker, 0.0)
        if shares_held <= _EPSILON or current_weight <= _EPSILON:
            return None, 0.0, [primary_reason, "no_position"]
        if remaining_turnover <= _EPSILON:
            return None, 0.0, [primary_reason, "turnover_limit"]

        sell_weight = min(float(desired_sell_weight), current_weight, remaining_turnover)
        if sell_weight <= _EPSILON:
            return None, 0.0, [primary_reason, "rebalance_threshold"]

        reasons = [primary_reason]
        if sell_weight < desired_sell_weight - _EPSILON:
            reasons.append("turnover_limit")

        if sell_weight >= current_weight - _EPSILON:
            return (
                Action(
                    action_type=ActionType.SELL,
                    ticker=ticker,
                    quantity_type=QuantityType.CLOSE_ALL,
                ),
                current_weight,
                reasons,
            )

        fraction_to_sell = min(1.0, sell_weight / current_weight)
        return (
            Action(
                action_type=ActionType.SELL,
                ticker=ticker,
                quantity=float(shares_held * fraction_to_sell),
                quantity_type=QuantityType.SHARES,
            ),
            sell_weight,
            reasons,
        )

    @staticmethod
    def _mark_action_limit(
        reasons_by_ticker: dict[str, list[str]],
        available_tickers: Sequence[str],
    ) -> None:
        for ticker in available_tickers:
            reasons_by_ticker.setdefault(ticker, []).append("max_actions_limit")

    def _log_decision(
        self,
        *,
        observation: Observation,
        signals: Sequence[TickerSignal],
        selected_tickers: Sequence[str],
        holdings_before: dict[str, dict[str, float]],
        target_weights: dict[str, float],
        estimated_post_decision_weights: dict[str, float],
        actions_and_reasons: Sequence[tuple[Action, tuple[str, ...]]],
        reasons_by_ticker: dict[str, tuple[str, ...]],
        remaining_cash_weight_estimate: float,
        remaining_turnover_budget: float,
    ) -> None:
        action_payloads = tuple(
            self._action_payload(action, reasons)
            for action, reasons in actions_and_reasons
        )
        record = _DecisionRecord(
            week_index=int(observation.week_index),
            date=observation.date,
            candidate_scores=tuple(signal.to_dict() for signal in signals),
            selected_tickers=tuple(selected_tickers),
            holdings_before=holdings_before,
            target_weights={ticker: float(value) for ticker, value in sorted(target_weights.items())},
            estimated_post_decision_weights={
                ticker: float(value)
                for ticker, value in sorted(estimated_post_decision_weights.items())
            },
            generated_actions=action_payloads,
            reasons_by_ticker={
                ticker: tuple(reasons)
                for ticker, reasons in sorted(reasons_by_ticker.items())
            },
            remaining_cash_weight_estimate=float(remaining_cash_weight_estimate),
            remaining_turnover_budget=float(remaining_turnover_budget),
        )
        self._decision_records.append(record)

    @staticmethod
    def _action_payload(action: Action, reasons: Sequence[str]) -> dict[str, Any]:
        return {
            "action_type": action.action_type.value,
            "ticker": action.ticker,
            "quantity": action.quantity,
            "quantity_type": action.quantity_type.value if action.quantity_type else None,
            "fraction": action.fraction,
            "stop_price": action.stop_price,
            "reasons": tuple(reasons),
        }

    @classmethod
    def _json_ready(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(key): cls._json_ready(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_ready(item) for item in value]
        return value
