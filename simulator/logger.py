"""Structured audit logging for deterministic weekly simulation runs.

The logger stores one :class:`BatchLogEntry` per :meth:`TradingEnvironment.step`
call and keeps all nested action-, stop-, and gap-adjustment detail inside that
batch. These logs are post-step audit artifacts: they may include execution and
mark-to-market details from the week that has just completed, and are therefore
not a substitute for decision-time visible state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .actions import Action, ExecutionResult, ValidationOutcome, ValidationResult
from .observation import PendingLiquidation
from .state import PortfolioState


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """Compact immutable portfolio snapshot stored at batch end."""

    cash: float
    total_nav: float
    realized_pnl: float
    weekly_turnover: float
    concentration_hhi: float
    portfolio_volatility: Optional[float]
    shares: tuple[tuple[str, float], ...]
    market_value: tuple[tuple[str, float], ...]
    stop_levels: tuple[tuple[str, float], ...]

    @classmethod
    def from_state(cls, state: PortfolioState) -> "PortfolioSnapshot":
        """Build a batch-end snapshot from a frozen portfolio state."""
        return cls(
            cash=float(state.cash),
            total_nav=float(state.total_nav),
            realized_pnl=float(state.realized_pnl),
            weekly_turnover=float(state.weekly_turnover),
            concentration_hhi=float(state.concentration_hhi),
            portfolio_volatility=(
                None
                if state.portfolio_volatility is None
                else float(state.portfolio_volatility)
            ),
            shares=state.shares,
            market_value=state.market_value,
            stop_levels=state.stop_levels,
        )


@dataclass(frozen=True, slots=True)
class ActionLogEntry:
    """One logged action outcome inside a weekly batch.

    ``action_source`` distinguishes user-submitted actions from internal
    simulator executions such as staged stop-loss liquidations.
    """

    processing_order: int
    action_index: Optional[int]
    action_source: str
    is_internal: bool
    is_imputed: bool
    action_type: str
    ticker: Optional[str]
    quantity: Optional[float]
    quantity_type: Optional[str]
    fraction: Optional[float]
    stop_price: Optional[float]
    effective_quantity: Optional[float]
    effective_quantity_type: Optional[str]
    effective_fraction: Optional[float]
    effective_stop_price: Optional[float]
    validation_outcome: Optional[str]
    validation_reason: Optional[str]
    clip_delta: Optional[float]
    vol_rule_active: Optional[bool]
    rule_triggered: Optional[str]
    executed_shares: Optional[float]
    execution_price: Optional[float]
    gross_trade_value: Optional[float]
    total_cost: Optional[float]
    commission_cost: Optional[float]
    spread_cost: Optional[float]
    slippage_cost: Optional[float]
    week_executed: Optional[int]
    is_stop_loss: bool
    gap_adjusted: bool
    execution_status: str
    execution_reason: Optional[str]
    stop_action_applied: Optional[bool]
    stop_action_status: Optional[str]

    @classmethod
    def from_records(
        cls,
        *,
        processing_order: int,
        action_index: Optional[int],
        action_source: str,
        action: Action,
        validation_result: ValidationResult | None = None,
        execution_result: ExecutionResult | None = None,
        is_internal: bool = False,
        is_imputed: bool = False,
        validation_outcome_override: str | None = None,
        validation_reason_override: str | None = None,
        rule_triggered_override: str | None = None,
        gap_adjusted_override: bool | None = None,
        execution_status_override: str | None = None,
        execution_reason_override: str | None = None,
        stop_action_applied: bool | None = None,
        stop_action_status: str | None = None,
    ) -> "ActionLogEntry":
        """Construct a log entry from the public simulator DTOs."""
        effective_action = (
            validation_result.effective_action
            if validation_result is not None and validation_result.effective_action is not None
            else action
        )

        return cls(
            processing_order=processing_order,
            action_index=action_index,
            action_source=action_source,
            is_internal=is_internal,
            is_imputed=is_imputed,
            action_type=action.action_type.value,
            ticker=action.ticker,
            quantity=action.quantity,
            quantity_type=(
                action.quantity_type.value if action.quantity_type is not None else None
            ),
            fraction=action.fraction,
            stop_price=action.stop_price,
            effective_quantity=effective_action.quantity,
            effective_quantity_type=(
                effective_action.quantity_type.value
                if effective_action.quantity_type is not None
                else None
            ),
            effective_fraction=effective_action.fraction,
            effective_stop_price=effective_action.stop_price,
            validation_outcome=(
                validation_outcome_override
                if validation_outcome_override is not None
                else (
                    validation_result.outcome.value
                    if validation_result is not None
                    else ValidationOutcome.ACCEPTED.value
                )
            ),
            validation_reason=(
                validation_reason_override
                if validation_reason_override is not None
                else (validation_result.reason if validation_result is not None else None)
            ),
            clip_delta=validation_result.clip_delta if validation_result is not None else None,
            vol_rule_active=(
                validation_result.vol_rule_active if validation_result is not None else None
            ),
            rule_triggered=(
                rule_triggered_override
                if rule_triggered_override is not None
                else (
                    validation_result.rule_triggered
                    if validation_result is not None
                    else None
                )
            ),
            executed_shares=(
                float(execution_result.executed_shares)
                if execution_result is not None
                else None
            ),
            execution_price=(
                float(execution_result.execution_price)
                if execution_result is not None
                else None
            ),
            gross_trade_value=(
                float(execution_result.gross_trade_value)
                if execution_result is not None
                else None
            ),
            total_cost=(
                float(execution_result.total_cost)
                if execution_result is not None
                else None
            ),
            commission_cost=(
                float(execution_result.commission_cost)
                if execution_result is not None
                else None
            ),
            spread_cost=(
                float(execution_result.spread_cost)
                if execution_result is not None
                else None
            ),
            slippage_cost=(
                float(execution_result.slippage_cost)
                if execution_result is not None
                else None
            ),
            week_executed=execution_result.week_executed if execution_result is not None else None,
            is_stop_loss=bool(execution_result.is_stop_loss) if execution_result is not None else False,
            gap_adjusted=(
                bool(gap_adjusted_override)
                if gap_adjusted_override is not None
                else (bool(execution_result.gap_adjusted) if execution_result is not None else False)
            ),
            execution_status=(
                execution_status_override
                if execution_status_override is not None
                else ("executed" if execution_result is not None else "not_applicable")
            ),
            execution_reason=execution_reason_override,
            stop_action_applied=stop_action_applied,
            stop_action_status=stop_action_status,
        )


@dataclass(frozen=True, slots=True)
class StopTriggerLogEntry:
    """One stop-loss breach detected after a completed week."""

    ticker: str
    triggered_by_low: float
    stop_level: float
    execution_week: Optional[int]
    staged: bool
    reason: Optional[str] = None


@dataclass(frozen=True, slots=True)
class GapAdjustmentLogEntry:
    """One execution-time clip caused by open-vs-close cash drift."""

    action_index: Optional[int]
    ticker: str
    original_shares: float
    adjusted_shares: float
    delta_shares: float
    original_gross_trade_value: float
    adjusted_gross_trade_value: float
    reason: str


@dataclass(frozen=True, slots=True)
class PendingLiquidationLogEntry:
    """A pending staged liquidation known to the environment at batch boundaries."""

    ticker: str
    triggered_by_low: float
    stop_level: float
    execution_week: int

    @classmethod
    def from_pending(cls, pending: PendingLiquidation) -> "PendingLiquidationLogEntry":
        """Convert a live pending liquidation DTO into a log entry."""
        return cls(
            ticker=pending.ticker,
            triggered_by_low=float(pending.triggered_by_low),
            stop_level=float(pending.stop_level),
            execution_week=int(pending.execution_week),
        )


@dataclass(frozen=True, slots=True)
class BatchLogEntry:
    """One deterministic audit record for a full weekly ``env.step()`` batch."""

    week_index: int
    date: datetime
    execution_week: int
    end_date: datetime
    batch_start_nav: float
    batch_end_nav: float
    n_actions_submitted: int
    n_accepted: int
    n_clipped: int
    n_rejected: int
    portfolio_snapshot: PortfolioSnapshot
    pending_liquidations_before: tuple[PendingLiquidationLogEntry, ...] = ()
    pending_liquidations_after: tuple[PendingLiquidationLogEntry, ...] = ()
    action_entries: tuple[ActionLogEntry, ...] = ()
    stop_trigger_entries: tuple[StopTriggerLogEntry, ...] = ()
    gap_adjustment_entries: tuple[GapAdjustmentLogEntry, ...] = ()


class AuditLogger:
    """Accumulate batch logs and export reproducible audit artifacts."""

    _ACTION_COLUMNS: tuple[str, ...] = (
        "week_index",
        "date",
        "execution_week",
        "end_date",
        "batch_start_nav",
        "batch_end_nav",
        "n_actions_submitted",
        "n_accepted",
        "n_clipped",
        "n_rejected",
        "batch_cash",
        "batch_total_nav",
        "batch_realized_pnl",
        "batch_weekly_turnover",
        "batch_concentration_hhi",
        "batch_portfolio_volatility",
        "processing_order",
        "action_index",
        "action_source",
        "is_internal",
        "is_imputed",
        "action_type",
        "ticker",
        "quantity",
        "quantity_type",
        "fraction",
        "stop_price",
        "effective_quantity",
        "effective_quantity_type",
        "effective_fraction",
        "effective_stop_price",
        "validation_outcome",
        "validation_reason",
        "clip_delta",
        "vol_rule_active",
        "rule_triggered",
        "executed_shares",
        "execution_price",
        "gross_trade_value",
        "total_cost",
        "commission_cost",
        "spread_cost",
        "slippage_cost",
        "week_executed",
        "is_stop_loss",
        "gap_adjusted",
        "execution_status",
        "execution_reason",
        "stop_action_applied",
        "stop_action_status",
    )
    _BATCH_COLUMNS: tuple[str, ...] = (
        "week_index",
        "date",
        "execution_week",
        "end_date",
        "batch_start_nav",
        "batch_end_nav",
        "n_actions_submitted",
        "n_accepted",
        "n_clipped",
        "n_rejected",
        "cash",
        "total_nav",
        "realized_pnl",
        "weekly_turnover",
        "concentration_hhi",
        "portfolio_volatility",
        "shares",
        "market_value",
        "stop_levels",
        "pending_liquidations_before",
        "pending_liquidations_after",
        "n_pending_liquidations_before",
        "n_pending_liquidations_after",
        "n_action_entries",
        "n_stop_triggers",
        "n_gap_adjustments",
    )
    _STOP_TRIGGER_COLUMNS: tuple[str, ...] = (
        "week_index",
        "date",
        "execution_week",
        "end_date",
        "ticker",
        "triggered_by_low",
        "stop_level",
        "execution_week_scheduled",
        "staged",
        "reason",
    )
    _GAP_ADJUSTMENT_COLUMNS: tuple[str, ...] = (
        "week_index",
        "date",
        "execution_week",
        "end_date",
        "action_index",
        "ticker",
        "original_shares",
        "adjusted_shares",
        "delta_shares",
        "original_gross_trade_value",
        "adjusted_gross_trade_value",
        "reason",
    )

    def __init__(self) -> None:
        self._entries: list[BatchLogEntry] = []

    def clear(self) -> None:
        """Remove all accumulated batch entries."""
        self._entries.clear()

    def log_batch(self, batch_entry: BatchLogEntry) -> None:
        """Append one completed batch log entry."""
        if not isinstance(batch_entry, BatchLogEntry):
            raise TypeError("batch_entry must be a BatchLogEntry")
        self._entries.append(batch_entry)

    @property
    def entries(self) -> tuple[BatchLogEntry, ...]:
        """Immutable view of accumulated batch entries."""
        return tuple(self._entries)

    def to_action_dataframe(self, include_internal: bool = True) -> pd.DataFrame:
        """Return one flattened row per action entry.

        Batch-level numeric fields are repeated for each action so the result
        can be exported directly to CSV without additional joins.
        """
        rows: list[dict[str, Any]] = []
        for batch in self._entries:
            base = self._batch_base_row(batch)
            for entry in batch.action_entries:
                if not include_internal and entry.is_internal:
                    continue
                row = dict(base)
                row.update(asdict(entry))
                rows.append(row)
        frame = pd.DataFrame(rows, columns=self._ACTION_COLUMNS)
        if frame.empty:
            return frame
        return frame.sort_values(
            by=["week_index", "processing_order", "action_index"],
            na_position="last",
        ).reset_index(drop=True)

    def to_batch_dataframe(self) -> pd.DataFrame:
        """Return one row per logged weekly batch."""
        rows: list[dict[str, Any]] = []
        for batch in self._entries:
            snapshot = batch.portfolio_snapshot
            rows.append(
                {
                    "week_index": batch.week_index,
                    "date": batch.date,
                    "execution_week": batch.execution_week,
                    "end_date": batch.end_date,
                    "batch_start_nav": batch.batch_start_nav,
                    "batch_end_nav": batch.batch_end_nav,
                    "n_actions_submitted": batch.n_actions_submitted,
                    "n_accepted": batch.n_accepted,
                    "n_clipped": batch.n_clipped,
                    "n_rejected": batch.n_rejected,
                    "cash": snapshot.cash,
                    "total_nav": snapshot.total_nav,
                    "realized_pnl": snapshot.realized_pnl,
                    "weekly_turnover": snapshot.weekly_turnover,
                    "concentration_hhi": snapshot.concentration_hhi,
                    "portfolio_volatility": snapshot.portfolio_volatility,
                    "shares": self._mapping_json(snapshot.shares),
                    "market_value": self._mapping_json(snapshot.market_value),
                    "stop_levels": self._mapping_json(snapshot.stop_levels),
                    "pending_liquidations_before": self._pending_liquidations_json(
                        batch.pending_liquidations_before
                    ),
                    "pending_liquidations_after": self._pending_liquidations_json(
                        batch.pending_liquidations_after
                    ),
                    "n_pending_liquidations_before": len(batch.pending_liquidations_before),
                    "n_pending_liquidations_after": len(batch.pending_liquidations_after),
                    "n_action_entries": len(batch.action_entries),
                    "n_stop_triggers": len(batch.stop_trigger_entries),
                    "n_gap_adjustments": len(batch.gap_adjustment_entries),
                }
            )
        frame = pd.DataFrame(rows, columns=self._BATCH_COLUMNS)
        if frame.empty:
            return frame
        return frame.sort_values(by=["week_index"]).reset_index(drop=True)

    def to_stop_trigger_dataframe(self) -> pd.DataFrame:
        """Return one flattened row per stop trigger."""
        rows: list[dict[str, Any]] = []
        for batch in self._entries:
            base = {
                "week_index": batch.week_index,
                "date": batch.date,
                "execution_week": batch.execution_week,
                "end_date": batch.end_date,
            }
            for entry in batch.stop_trigger_entries:
                row = dict(base)
                row.update(
                    {
                        "ticker": entry.ticker,
                        "triggered_by_low": entry.triggered_by_low,
                        "stop_level": entry.stop_level,
                        "execution_week_scheduled": entry.execution_week,
                        "staged": entry.staged,
                        "reason": entry.reason,
                    }
                )
                rows.append(row)
        frame = pd.DataFrame(rows, columns=self._STOP_TRIGGER_COLUMNS)
        if frame.empty:
            return frame
        return frame.sort_values(
            by=["week_index", "execution_week", "ticker"],
            na_position="last",
        ).reset_index(drop=True)

    def to_gap_adjustment_dataframe(self) -> pd.DataFrame:
        """Return one flattened row per execution-time gap adjustment."""
        rows: list[dict[str, Any]] = []
        for batch in self._entries:
            base = {
                "week_index": batch.week_index,
                "date": batch.date,
                "execution_week": batch.execution_week,
                "end_date": batch.end_date,
            }
            for entry in batch.gap_adjustment_entries:
                row = dict(base)
                row.update(asdict(entry))
                rows.append(row)
        frame = pd.DataFrame(rows, columns=self._GAP_ADJUSTMENT_COLUMNS)
        if frame.empty:
            return frame
        return frame.sort_values(
            by=["week_index", "action_index", "ticker"],
            na_position="last",
        ).reset_index(drop=True)

    def get_action_log(self) -> pd.DataFrame:
        """Compatibility wrapper returning the full action log."""
        return self.to_action_dataframe(include_internal=True)

    def get_validation_log(self) -> pd.DataFrame:
        """Return user-submitted validation outcomes only."""
        action_df = self.to_action_dataframe(include_internal=False)
        if action_df.empty:
            return pd.DataFrame(
                columns=[
                    "week_index",
                    "date",
                    "processing_order",
                    "action_index",
                    "action_type",
                    "ticker",
                    "validation_outcome",
                    "validation_reason",
                    "clip_delta",
                    "vol_rule_active",
                    "rule_triggered",
                    "is_imputed",
                    "execution_status",
                    "execution_reason",
                    "stop_action_applied",
                    "stop_action_status",
                ]
            )
        return action_df.loc[
            :,
            [
                "week_index",
                "date",
                "processing_order",
                "action_index",
                "action_type",
                "ticker",
                "validation_outcome",
                "validation_reason",
                "clip_delta",
                "vol_rule_active",
                "rule_triggered",
                "is_imputed",
                "execution_status",
                "execution_reason",
                "stop_action_applied",
                "stop_action_status",
            ],
        ].copy()

    def get_execution_log(self, include_internal: bool = True) -> pd.DataFrame:
        """Return executed trades only."""
        action_df = self.to_action_dataframe(include_internal=include_internal)
        if action_df.empty or "executed_shares" not in action_df.columns:
            return pd.DataFrame(columns=self._ACTION_COLUMNS)
        return action_df[action_df["executed_shares"].notna()].reset_index(drop=True).copy()

    def export_csv(self, path: str | Path) -> None:
        """Export the flattened action log as CSV."""
        output_path = Path(path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_action_dataframe(include_internal=True).to_csv(output_path, index=False)

    def export_jsonl(self, path: str | Path) -> None:
        """Export the nested batch log as JSONL, one batch per line."""
        output_path = Path(path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for entry in self._entries:
                payload = self._json_ready(asdict(entry))
                handle.write(json.dumps(payload, sort_keys=True))
                handle.write("\n")

    def _batch_base_row(self, batch: BatchLogEntry) -> dict[str, Any]:
        snapshot = batch.portfolio_snapshot
        return {
            "week_index": batch.week_index,
            "date": batch.date,
            "execution_week": batch.execution_week,
            "end_date": batch.end_date,
            "batch_start_nav": batch.batch_start_nav,
            "batch_end_nav": batch.batch_end_nav,
            "n_actions_submitted": batch.n_actions_submitted,
            "n_accepted": batch.n_accepted,
            "n_clipped": batch.n_clipped,
            "n_rejected": batch.n_rejected,
            "batch_cash": snapshot.cash,
            "batch_total_nav": snapshot.total_nav,
            "batch_realized_pnl": snapshot.realized_pnl,
            "batch_weekly_turnover": snapshot.weekly_turnover,
            "batch_concentration_hhi": snapshot.concentration_hhi,
            "batch_portfolio_volatility": snapshot.portfolio_volatility,
        }

    @staticmethod
    def _mapping_json(pairs: tuple[tuple[str, float], ...]) -> str:
        return json.dumps(dict(pairs), sort_keys=True)

    @staticmethod
    def _pending_liquidations_json(
        entries: tuple[PendingLiquidationLogEntry, ...],
    ) -> str:
        payload = [asdict(entry) for entry in entries]
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def _json_ready(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(key): cls._json_ready(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_ready(item) for item in value]
        return value
