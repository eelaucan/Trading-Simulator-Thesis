"""Main weekly trading environment.

This orchestrator wires the Stage 1 and Stage 2 building blocks into one
usable simulation loop while preserving three key invariants:

1. Only :class:`Observation` is visible to the decision-maker.
2. All state transitions remain immutable.
3. No future market rows are exposed through the observation surface.

Timing note
-----------
One ``step()`` spans the transition from the end of decision week ``t`` to the
end of week ``t + 1``. For that reason, stop-triggered pending liquidations
scheduled for week ``t + 1`` are projected during validation and then executed
at ``open[t + 1]`` before user-submitted trades at the same open.

The open ordering rule is therefore explicit and deterministic:
previously scheduled liquidations execute first, then newly submitted user
trades for decision week ``t`` execute afterward at that same ``open[t + 1]``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Sequence

from .actions import (
    Action,
    ActionType,
    ExecutionResult,
    QuantityType,
    ValidationOutcome,
    ValidationResult,
)
from .config import SimulatorConfig
from .execution import ExecutionEngine, ExecutionEstimate, NonExecutableTradeError
from .logger import (
    ActionLogEntry,
    AuditLogger,
    BatchLogEntry,
    GapAdjustmentLogEntry,
    PendingLiquidationLogEntry,
    PortfolioSnapshot,
    StopTriggerLogEntry,
)
from .market import MarketReplay
from .metrics import MetricsEngine, SimulationMetrics
from .observation import Observation, PendingLiquidation
from .portfolio import PortfolioManager
from .state import PortfolioState
from .validator import ConstraintValidator


_EPSILON = 1e-12
_ACTION_ORDER: dict[ActionType, int] = {
    ActionType.SELL: 0,
    ActionType.REDUCE: 0,
    ActionType.BUY: 1,
    ActionType.SET_STOP: 2,
    ActionType.REMOVE_STOP: 2,
    ActionType.HOLD: 3,
}


@dataclass(frozen=True, slots=True)
class _SubmittedAction:
    """Internal normalized action wrapper for one weekly batch."""

    action: Action
    action_index: int
    is_imputed: bool = False


class TradingEnvironment:
    """Weekly deterministic simulation environment.

    ``step(actions)`` uses the current observation at week ``t`` as the
    decision boundary, then internally advances the simulator through the
    ``t + 1`` open and ``t + 1`` close. The returned observation is therefore
    the next decision-time view at week ``t + 1``.
    """

    def __init__(
        self,
        market: MarketReplay,
        config: SimulatorConfig | None = None,
        executor: ExecutionEngine | None = None,
        validator: ConstraintValidator | None = None,
        portfolio_manager: PortfolioManager | None = None,
        logger: AuditLogger | None = None,
        metrics_engine: MetricsEngine | None = None,
    ) -> None:
        base_config = config or SimulatorConfig(ticker_universe=market.available_tickers)
        self._market = market
        self._config = base_config
        self._ticker_universe = self._resolve_ticker_universe(base_config, market)
        self._executor = executor or ExecutionEngine(base_config)
        self._portfolio_manager = portfolio_manager or PortfolioManager(base_config)
        self._validator = validator or ConstraintValidator(base_config, self._executor)
        self._logger = logger or AuditLogger()
        self._metrics_engine = metrics_engine or MetricsEngine(base_config)

        self._state: PortfolioState | None = None
        self._observation: Observation | None = None
        self._current_week: int = 0
        self._initial_decision_week: int = 0
        self._pending_liquidations: list[PendingLiquidation] = []
        self._done: bool = False

    def reset(self) -> tuple[Observation, PortfolioState]:
        """Reset the run to the first decision week and return visible state.

        The environment does not have to begin at raw market week ``0``. By
        default it starts at the earliest actionable week that exposes a full
        ``observation_history_weeks`` window, so a participant can see
        meaningful historical context before taking the first action. This
        preserves no-look-ahead because the initial observation still contains
        only rows with ``_week_idx <= initial_decision_week``.
        """
        initial_decision_week = self._resolve_initial_decision_week()
        start_date = self._market.get_date(initial_decision_week)
        initial_state = self._portfolio_manager.initialize(
            week_index=initial_decision_week,
            date=start_date,
            ticker_universe=self._ticker_universe,
        )
        self._logger.clear()
        self._pending_liquidations = []
        self._state = initial_state
        self._initial_decision_week = initial_decision_week
        self._current_week = initial_decision_week
        self._done = self._current_week >= self._market.n_weeks - 1
        self._observation = self._build_observation(
            week_index=initial_decision_week,
            state=initial_state,
            pending_liquidations=[],
        )
        return self._observation, initial_state

    def step(
        self,
        actions: Sequence[Action],
    ) -> tuple[Observation, PortfolioState, bool, dict[str, Any]]:
        """Advance one weekly decision event.

        The returned ``info`` payload is post-step internal telemetry. It may
        contain execution and stop-trigger details from the just-completed
        week, but it never changes the no-look-ahead boundary enforced by the
        returned :class:`Observation`.
        """
        if self._state is None or self._observation is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Simulation is finished. Call reset() to start a new run.")

        t = self._current_week
        execution_week = t + 1
        if execution_week >= self._market.n_weeks:
            raise RuntimeError("No next market week is available for execution.")

        submitted_actions = list(actions)
        if len(submitted_actions) > self._config.max_actions_per_step:
            raise ValueError(
                f"Received {len(submitted_actions)} actions, but max_actions_per_step="
                f"{self._config.max_actions_per_step}"
            )

        current_state = self._state
        current_observation = self._build_observation(
            week_index=t,
            state=current_state,
            pending_liquidations=self._pending_liquidations,
        )
        pending_before = tuple(
            PendingLiquidationLogEntry.from_pending(item)
            for item in sorted(
                self._pending_liquidations,
                key=lambda item: (item.execution_week, item.ticker),
            )
        )
        normalized_actions = self._normalize_actions(submitted_actions)
        batch_start_nav = float(current_state.total_nav)

        due_pending = self._pending_liquidations_due_for_execution(execution_week)
        projected_state = current_state
        accumulated_turnover_dollars = 0.0

        projected_state, accumulated_turnover_dollars = self._project_pending_liquidations(
            state=projected_state,
            decision_week=t,
            batch_start_nav=batch_start_nav,
            accumulated_turnover_dollars=accumulated_turnover_dollars,
            due_pending=due_pending,
        )

        validation_results: list[ValidationResult] = []
        for submitted in normalized_actions:
            action = submitted.action
            estimate = self._executor.estimate_cost(
                action=action,
                state=projected_state,
                t=t,
                market=self._market,
                batch_start_nav=batch_start_nav,
            )
            validation_result = self._validator.validate(
                action=action,
                projected_state=projected_state,
                market=self._market,
                t=t,
                accumulated_turnover_dollars=accumulated_turnover_dollars,
                batch_start_nav=batch_start_nav,
                estimated_cost=estimate,
            )
            validation_results.append(validation_result)

            if (
                validation_result.outcome in {ValidationOutcome.ACCEPTED, ValidationOutcome.CLIPPED}
                and validation_result.effective_action is not None
            ):
                projected_state, accumulated_turnover_dollars = self._project_effective_action(
                    state=projected_state,
                    action=validation_result.effective_action,
                    decision_week=t,
                    batch_start_nav=batch_start_nav,
                    accumulated_turnover_dollars=accumulated_turnover_dollars,
                )

        live_state = current_state
        accumulated_gross_executed = 0.0
        internal_action_entries: list[ActionLogEntry] = []
        pending_execution_results: list[ExecutionResult] = []
        execution_skip_messages: list[str] = []
        execution_note_by_index: dict[int, str] = {}
        remaining_pending = [
            pending
            for pending in self._pending_liquidations
            if pending.execution_week != execution_week
        ]

        # Previously staged liquidations always take priority at the next open.
        for pending in due_pending:
            shares_held = live_state.shares_dict().get(pending.ticker, 0.0)
            if self._executor.is_effectively_zero_shares(shares_held):
                continue
            liquidation_action = Action(
                action_type=ActionType.SELL,
                ticker=pending.ticker,
                quantity_type=QuantityType.CLOSE_ALL,
            )
            try:
                execution_result = self._executor.execute(
                    action=liquidation_action,
                    t=t,
                    market=self._market,
                    state=live_state,
                    batch_start_nav=batch_start_nav,
                )
            except NonExecutableTradeError:
                continue
            execution_result = dataclasses.replace(
                execution_result,
                is_stop_loss=True,
                original_shares=abs(execution_result.executed_shares),
            )
            live_state = self._portfolio_manager.apply_execution(
                execution_result,
                live_state,
                batch_start_nav=batch_start_nav,
                accumulated_gross_traded=accumulated_gross_executed,
            )
            accumulated_gross_executed += float(execution_result.gross_trade_value)
            pending_execution_results.append(execution_result)
            internal_action_entries.append(
                ActionLogEntry.from_records(
                    processing_order=len(internal_action_entries),
                    action_index=None,
                    action_source="pending_liquidation",
                    action=liquidation_action,
                    execution_result=execution_result,
                    is_internal=True,
                    validation_outcome_override=ValidationOutcome.ACCEPTED.value,
                    validation_reason_override="Pending liquidation executed at scheduled open",
                    rule_triggered_override="pending_liquidation",
                    execution_status_override="executed",
                )
            )

        execution_results: list[ExecutionResult | None] = []
        gap_adjustment_entries: list[GapAdjustmentLogEntry] = []
        gap_adjusted_indices: set[int] = set()
        gap_clipped_to_zero_indices: set[int] = set()

        for submitted, validation_result in zip(normalized_actions, validation_results):
            effective_action = validation_result.effective_action
            if (
                validation_result.outcome == ValidationOutcome.REJECTED
                or effective_action is None
                or effective_action.action_type not in {ActionType.BUY, ActionType.SELL, ActionType.REDUCE}
            ):
                execution_results.append(None)
                continue

            execution_action = effective_action
            if execution_action.action_type == ActionType.BUY:
                execution_action, gap_entry = self._maybe_gap_adjust_buy_action(
                    action=execution_action,
                    state=live_state,
                    decision_week=t,
                    batch_start_nav=batch_start_nav,
                    action_index=submitted.action_index,
                )
                if gap_entry is not None:
                    gap_adjustment_entries.append(gap_entry)
                    gap_adjusted_indices.add(submitted.action_index)
                if execution_action is None:
                    gap_clipped_to_zero_indices.add(submitted.action_index)
                    skip_reason = (
                        "A planned buy became too small after the next open price and estimated costs were applied, so it was skipped."
                    )
                    execution_note_by_index[submitted.action_index] = skip_reason
                    execution_skip_messages.append(skip_reason)
                    execution_results.append(None)
                    continue

            execution_action, execution_adjustment_reason = self._adjust_action_for_live_execution(
                action=execution_action,
                state=live_state,
                decision_week=t,
                batch_start_nav=batch_start_nav,
            )
            if execution_adjustment_reason is not None:
                execution_note_by_index[submitted.action_index] = execution_adjustment_reason
            if execution_action is None:
                if execution_adjustment_reason is not None:
                    execution_skip_messages.append(execution_adjustment_reason)
                execution_results.append(None)
                continue

            pre_execution_skip_reason = self._pre_execution_skip_reason(
                action=execution_action,
                state=live_state,
                decision_week=t,
                batch_start_nav=batch_start_nav,
            )
            if pre_execution_skip_reason is not None:
                execution_note_by_index[submitted.action_index] = pre_execution_skip_reason
                execution_skip_messages.append(pre_execution_skip_reason)
                execution_results.append(None)
                continue

            try:
                execution_result = self._executor.execute(
                    action=execution_action,
                    t=t,
                    market=self._market,
                    state=live_state,
                    batch_start_nav=batch_start_nav,
                )
            except NonExecutableTradeError:
                skip_reason = self._default_zero_size_skip_reason(execution_action)
                execution_note_by_index[submitted.action_index] = skip_reason
                execution_skip_messages.append(skip_reason)
                execution_results.append(None)
                continue
            if submitted.action_index in gap_adjusted_indices:
                original_shares = next(
                    entry.original_shares
                    for entry in reversed(gap_adjustment_entries)
                    if entry.action_index == submitted.action_index
                )
                execution_result = dataclasses.replace(
                    execution_result,
                    gap_adjusted=True,
                    original_shares=original_shares,
                )

            live_state = self._portfolio_manager.apply_execution(
                execution_result,
                live_state,
                batch_start_nav=batch_start_nav,
                accumulated_gross_traded=accumulated_gross_executed,
            )
            accumulated_gross_executed += float(execution_result.gross_trade_value)
            execution_results.append(execution_result)

        stop_action_status_by_index: dict[int, tuple[bool | None, str | None]] = {}
        skipped_stop_actions: list[tuple[int, str, str]] = []
        for submitted, validation_result in zip(normalized_actions, validation_results):
            effective_action = validation_result.effective_action
            if (
                validation_result.outcome not in {ValidationOutcome.ACCEPTED, ValidationOutcome.CLIPPED}
                or effective_action is None
                or effective_action.action_type not in {ActionType.SET_STOP, ActionType.REMOVE_STOP}
            ):
                continue

            if effective_action.action_type == ActionType.SET_STOP:
                ticker = effective_action.ticker or ""
                if live_state.shares_dict().get(ticker, 0.0) <= _EPSILON:
                    stop_action_status_by_index[submitted.action_index] = (
                        False,
                        "skipped_no_position_after_execution",
                    )
                    skipped_stop_actions.append(
                        (submitted.action_index, ticker, "skipped_no_position_after_execution")
                    )
                    continue
                live_state = self._portfolio_manager.update_stop(effective_action, live_state)
                stop_action_status_by_index[submitted.action_index] = (True, "applied")
            else:
                assert effective_action.ticker is not None
                if effective_action.ticker not in live_state.stop_levels_dict():
                    stop_action_status_by_index[submitted.action_index] = (
                        False,
                        "skipped_no_active_stop_after_execution",
                    )
                    skipped_stop_actions.append(
                        (
                            submitted.action_index,
                            effective_action.ticker,
                            "skipped_no_active_stop_after_execution",
                        )
                    )
                    continue
                live_state = self._portfolio_manager.remove_stop(effective_action.ticker, live_state)
                stop_action_status_by_index[submitted.action_index] = (True, "applied")

        close_prices = self._market.get_close_prices(execution_week)
        end_date = self._market.get_date(execution_week)
        live_state = self._portfolio_manager.mark_to_market(
            live_state,
            t=execution_week,
            date=end_date,
            close_prices=close_prices,
        )

        stop_trigger_entries: list[StopTriggerLogEntry] = []
        staged_pending: list[PendingLiquidation] = []
        triggered_liquidations = self._portfolio_manager.check_stop_triggers(
            live_state,
            execution_week,
            self._market,
        )
        for pending in triggered_liquidations:
            can_stage = pending.execution_week < self._market.n_weeks
            stop_trigger_entries.append(
                StopTriggerLogEntry(
                    ticker=pending.ticker,
                    triggered_by_low=float(pending.triggered_by_low),
                    stop_level=float(pending.stop_level),
                    execution_week=pending.execution_week if can_stage else None,
                    staged=can_stage,
                    reason=None if can_stage else "No later market week available for staged liquidation",
                )
            )
            live_state = self._portfolio_manager.remove_stop(pending.ticker, live_state)
            if can_stage:
                staged_pending.append(pending)

        self._pending_liquidations = sorted(
            remaining_pending + staged_pending,
            key=lambda item: (item.execution_week, item.ticker),
        )
        pending_after = tuple(
            PendingLiquidationLogEntry.from_pending(item)
            for item in self._pending_liquidations
        )

        n_accepted = sum(
            result.outcome == ValidationOutcome.ACCEPTED for result in validation_results
        )
        n_clipped = sum(
            result.outcome == ValidationOutcome.CLIPPED for result in validation_results
        )
        n_rejected = sum(
            result.outcome == ValidationOutcome.REJECTED for result in validation_results
        )

        user_action_entries = [
            ActionLogEntry.from_records(
                processing_order=len(internal_action_entries) + processing_order,
                action_index=submitted.action_index,
                action_source="user",
                action=submitted.action,
                validation_result=validation_result,
                execution_result=execution_result,
                is_internal=False,
                is_imputed=submitted.is_imputed,
                gap_adjusted_override=submitted.action_index in gap_adjusted_indices,
                execution_status_override=self._execution_status_for_action(
                    action=submitted.action,
                    validation_result=validation_result,
                    execution_result=execution_result,
                    gap_clipped_to_zero=submitted.action_index in gap_clipped_to_zero_indices,
                    execution_skip_reason=execution_note_by_index.get(submitted.action_index),
                ),
                execution_reason_override=execution_note_by_index.get(submitted.action_index),
                stop_action_applied=stop_action_status_by_index.get(submitted.action_index, (None, None))[0],
                stop_action_status=stop_action_status_by_index.get(submitted.action_index, (None, None))[1],
            )
            for processing_order, (submitted, validation_result, execution_result) in enumerate(zip(
                normalized_actions,
                validation_results,
                execution_results,
            ))
        ]

        batch_entry = BatchLogEntry(
            week_index=t,
            date=current_observation.date,
            execution_week=execution_week,
            end_date=end_date,
            batch_start_nav=batch_start_nav,
            batch_end_nav=float(live_state.total_nav),
            n_actions_submitted=len(submitted_actions),
            n_accepted=n_accepted,
            n_clipped=n_clipped,
            n_rejected=n_rejected,
            portfolio_snapshot=PortfolioSnapshot.from_state(live_state),
            pending_liquidations_before=pending_before,
            pending_liquidations_after=pending_after,
            action_entries=tuple(internal_action_entries + user_action_entries),
            stop_trigger_entries=tuple(stop_trigger_entries),
            gap_adjustment_entries=tuple(gap_adjustment_entries),
        )
        self._logger.log_batch(batch_entry)

        self._state = live_state
        self._current_week = execution_week
        blow_up_flag = self._metrics_engine.blow_up_check(live_state.nav_history)
        termination_reason: str | None = None
        if blow_up_flag:
            termination_reason = "blow_up"
        elif self._current_week >= self._market.n_weeks - 1:
            termination_reason = "end_of_data"
        self._done = termination_reason is not None
        self._observation = self._build_observation(
            week_index=execution_week,
            state=live_state,
            pending_liquidations=self._pending_liquidations,
        )

        info: dict[str, Any] = {
            "decision_week": t,
            "decision_date": current_observation.date,
            "execution_week": execution_week,
            "end_date": end_date,
            "batch_start_nav": batch_start_nav,
            "batch_end_nav": live_state.total_nav,
            "n_actions_submitted": len(submitted_actions),
            "n_accepted": n_accepted,
            "n_clipped": n_clipped,
            "n_rejected": n_rejected,
            "validation_results": tuple(validation_results),
            "execution_results": tuple(execution_results),
            "pending_liquidation_executions": tuple(pending_execution_results),
            "stop_triggers": tuple(stop_trigger_entries),
            "gap_adjustments": tuple(gap_adjustment_entries),
            "execution_skips": tuple(execution_skip_messages),
            "pending_liquidations_before": pending_before,
            "pending_liquidations_next": tuple(self._pending_liquidations),
            "skipped_stop_actions": tuple(skipped_stop_actions),
            "blow_up_flag": blow_up_flag,
            "termination_reason": termination_reason,
        }
        return self._observation, live_state, self._done, info

    def get_metrics(self) -> SimulationMetrics:
        """Compute final metrics from the current run state and audit log."""
        if self._state is None:
            raise RuntimeError("Call reset() before requesting metrics.")
        return self._metrics_engine.compute(self._state, self._logger)

    def compute_metrics(self) -> SimulationMetrics:
        """Compatibility alias for older code paths."""
        return self.get_metrics()

    @property
    def current_state(self) -> PortfolioState:
        """Current frozen portfolio state."""
        if self._state is None:
            raise RuntimeError("Environment has not been reset yet.")
        return self._state

    @property
    def current_observation(self) -> Observation:
        """Current visible observation."""
        if self._observation is None:
            raise RuntimeError("Environment has not been reset yet.")
        return self._observation

    @property
    def logger(self) -> AuditLogger:
        """Audit logger used by the environment."""
        return self._logger

    @property
    def config(self) -> SimulatorConfig:
        """Simulator configuration bound to this environment."""
        return self._config

    @property
    def done(self) -> bool:
        """Whether the episode has finished."""
        return self._done

    @property
    def initial_decision_week(self) -> int:
        """Resolved first decision week for the current episode."""
        return self._initial_decision_week

    def _build_observation(
        self,
        *,
        week_index: int,
        state: PortfolioState,
        pending_liquidations: Sequence[PendingLiquidation],
    ) -> Observation:
        """Build the visible state for a decision week."""
        return Observation(
            week_index=week_index,
            date=self._market.get_date(week_index),
            current_week_ohlcv=self._market.get_week_data(week_index),
            price_history=self._market.get_history(
                week_index,
                self._config.observation_history_weeks,
            ),
            portfolio_state=state,
            available_tickers=list(self._ticker_universe),
            pending_liquidations=list(pending_liquidations),
        )

    def _resolve_initial_decision_week(self) -> int:
        """Resolve the first actionable decision week for a new episode.

        The default behavior starts the participant late enough to expose a
        full visible history window, while still leaving at least one later
        week available for execution. If the dataset is shorter than that
        ideal window, the start is clamped to the last actionable week.
        """
        last_actionable_week = max(0, self._market.n_weeks - 2)
        configured_week = self._config.initial_decision_week

        if configured_week is None:
            desired_week = max(0, self._config.observation_history_weeks - 1)
            return min(desired_week, last_actionable_week)

        if configured_week > last_actionable_week:
            raise ValueError(
                "initial_decision_week must be <= the last actionable week "
                f"({last_actionable_week}) for a dataset with {self._market.n_weeks} week(s)"
            )
        return configured_week

    def _normalize_actions(self, actions: list[Action]) -> list[_SubmittedAction]:
        """Normalize an input batch into deterministic processing order."""
        if not actions:
            return [_SubmittedAction(action=Action(ActionType.HOLD), action_index=0, is_imputed=True)]

        has_hold = any(action.action_type == ActionType.HOLD for action in actions)
        if has_hold and len(actions) > 1:
            raise ValueError("HOLD cannot be mixed with other actions in the same batch")

        submitted = [
            _SubmittedAction(action=action, action_index=index, is_imputed=False)
            for index, action in enumerate(actions)
        ]
        return sorted(
            submitted,
            key=lambda item: (_ACTION_ORDER[item.action.action_type], item.action_index),
        )

    def _pending_liquidations_due_for_execution(
        self,
        execution_week: int,
    ) -> list[PendingLiquidation]:
        """Return staged liquidations due at the next execution open."""
        return sorted(
            (
                pending
                for pending in self._pending_liquidations
                if pending.execution_week == execution_week
            ),
            key=lambda item: item.ticker,
        )

    def _project_pending_liquidations(
        self,
        *,
        state: PortfolioState,
        decision_week: int,
        batch_start_nav: float,
        accumulated_turnover_dollars: float,
        due_pending: Sequence[PendingLiquidation],
    ) -> tuple[PortfolioState, float]:
        projected_state = state
        accumulated = accumulated_turnover_dollars

        for pending in due_pending:
            ticker = pending.ticker
            if projected_state.shares_dict().get(ticker, 0.0) <= _EPSILON:
                continue
            liquidation_action = Action(
                action_type=ActionType.SELL,
                ticker=ticker,
                quantity_type=QuantityType.CLOSE_ALL,
            )
            estimate = self._executor.estimate_cost(
                action=liquidation_action,
                state=projected_state,
                t=decision_week,
                market=self._market,
                batch_start_nav=batch_start_nav,
            )
            if self._executor.is_effectively_zero_shares(estimate.signed_shares):
                continue
            projected_state = self._apply_projected_trade(
                state=projected_state,
                action=liquidation_action,
                estimate=estimate,
                batch_start_nav=batch_start_nav,
                accumulated_gross_traded=accumulated,
                is_stop_loss=True,
            )
            accumulated += float(estimate.gross_trade_value)

        return projected_state, accumulated

    def _project_effective_action(
        self,
        *,
        state: PortfolioState,
        action: Action,
        decision_week: int,
        batch_start_nav: float,
        accumulated_turnover_dollars: float,
    ) -> tuple[PortfolioState, float]:
        if action.action_type == ActionType.HOLD:
            return state, accumulated_turnover_dollars

        if action.action_type in {ActionType.SET_STOP, ActionType.REMOVE_STOP}:
            if action.action_type == ActionType.SET_STOP:
                return self._portfolio_manager.update_stop(action, state), accumulated_turnover_dollars
            assert action.ticker is not None
            return (
                self._portfolio_manager.remove_stop(action.ticker, state),
                accumulated_turnover_dollars,
            )

        estimate = self._executor.estimate_cost(
            action=action,
            state=state,
            t=decision_week,
            market=self._market,
            batch_start_nav=batch_start_nav,
        )
        if self._executor.is_effectively_zero_shares(estimate.signed_shares):
            return state, accumulated_turnover_dollars

        projected_state = self._apply_projected_trade(
            state=state,
            action=action,
            estimate=estimate,
            batch_start_nav=batch_start_nav,
            accumulated_gross_traded=accumulated_turnover_dollars,
            is_stop_loss=False,
        )
        return projected_state, accumulated_turnover_dollars + float(estimate.gross_trade_value)

    def _apply_projected_trade(
        self,
        *,
        state: PortfolioState,
        action: Action,
        estimate: ExecutionEstimate,
        batch_start_nav: float,
        accumulated_gross_traded: float,
        is_stop_loss: bool,
    ) -> PortfolioState:
        assert action.ticker is not None
        projected_result = ExecutionResult(
            action=action,
            ticker=action.ticker,
            executed_shares=float(estimate.signed_shares),
            execution_price=float(estimate.reference_price),
            gross_trade_value=float(estimate.gross_trade_value),
            total_cost=float(estimate.total_cost),
            commission_cost=float(estimate.commission_cost),
            spread_cost=float(estimate.spread_cost),
            slippage_cost=float(estimate.slippage_cost),
            is_stop_loss=is_stop_loss,
        )
        return self._portfolio_manager.apply_execution(
            projected_result,
            state,
            batch_start_nav=batch_start_nav,
            accumulated_gross_traded=accumulated_gross_traded,
        )

    def _maybe_gap_adjust_buy_action(
        self,
        *,
        action: Action,
        state: PortfolioState,
        decision_week: int,
        batch_start_nav: float,
        action_index: int,
    ) -> tuple[Action | None, GapAdjustmentLogEntry | None]:
        assert action.ticker is not None
        execution_price = self._market.get_open_prices(decision_week + 1).get(action.ticker)
        if execution_price is None:
            raise ValueError(
                f"No open price available for {action.ticker!r} at week {decision_week + 1}"
            )

        desired_shares = self._executor.resolve_shares(
            action=action,
            reference_price=float(execution_price),
            state=state,
            batch_start_nav=batch_start_nav,
        )
        if self._executor.is_effectively_zero_shares(desired_shares):
            return None, None

        desired_estimate = self._executor.estimate_from_signed_shares(
            ticker=action.ticker,
            signed_shares=float(desired_shares),
            reference_price=float(execution_price),
            t=decision_week,
            market=self._market,
        )
        desired_cash_usage = float(desired_estimate.gross_trade_value + desired_estimate.total_cost)
        if desired_cash_usage <= state.cash + _EPSILON:
            return action, None

        low = 0.0
        high = float(desired_shares)
        for _ in range(60):
            mid = (low + high) / 2.0
            candidate = self._executor.estimate_from_signed_shares(
                ticker=action.ticker,
                signed_shares=mid,
                reference_price=float(execution_price),
                t=decision_week,
                market=self._market,
            )
            candidate_cash_usage = float(candidate.gross_trade_value + candidate.total_cost)
            if candidate_cash_usage <= state.cash + _EPSILON:
                low = mid
            else:
                high = mid

        adjusted_shares = low
        adjusted_estimate = self._executor.estimate_from_signed_shares(
            ticker=action.ticker,
            signed_shares=adjusted_shares,
            reference_price=float(execution_price),
            t=decision_week,
            market=self._market,
        )
        gap_entry = GapAdjustmentLogEntry(
            action_index=action_index,
            ticker=action.ticker,
            original_shares=float(desired_shares),
            adjusted_shares=float(adjusted_shares),
            delta_shares=float(desired_shares - adjusted_shares),
            original_gross_trade_value=float(desired_estimate.gross_trade_value),
            adjusted_gross_trade_value=float(adjusted_estimate.gross_trade_value),
            reason="open_gap_cash_shortfall",
        )

        if self._executor.is_effectively_zero_shares(adjusted_shares):
            return None, gap_entry

        adjusted_action = Action(
            action_type=ActionType.BUY,
            ticker=action.ticker,
            quantity=float(adjusted_shares),
            quantity_type=QuantityType.SHARES,
        )
        return adjusted_action, gap_entry

    def _adjust_action_for_live_execution(
        self,
        *,
        action: Action,
        state: PortfolioState,
        decision_week: int,
        batch_start_nav: float,
    ) -> tuple[Action | None, str | None]:
        """Apply final live-state execution guards before calling ``execute()``."""
        if action.action_type == ActionType.BUY:
            return action, None

        ticker = action.ticker
        if ticker is None:
            return action, None

        shares_held = float(state.shares_dict().get(ticker, 0.0))
        if self._executor.is_effectively_zero_shares(shares_held):
            return None, self._default_zero_size_skip_reason(action)

        if action.action_type == ActionType.REDUCE:
            return action, None

        if action.action_type == ActionType.SELL and action.quantity_type == QuantityType.CLOSE_ALL:
            return action, None

        execution_price = self._market.get_open_prices(decision_week + 1).get(ticker)
        if execution_price is None:
            return action, None

        desired_shares = abs(
            self._executor.resolve_shares(
                action=action,
                reference_price=float(execution_price),
                state=state,
                batch_start_nav=batch_start_nav,
            )
        )
        if self._executor.is_effectively_zero_shares(desired_shares):
            return None, self._default_zero_size_skip_reason(action)
        if desired_shares > shares_held + _EPSILON:
            return (
                Action(
                    action_type=ActionType.SELL,
                    ticker=ticker,
                    quantity_type=QuantityType.CLOSE_ALL,
                ),
                "A planned sale was clipped to the remaining live shares available at the next open.",
            )
        return action, None

    def _pre_execution_skip_reason(
        self,
        *,
        action: Action,
        state: PortfolioState,
        decision_week: int,
        batch_start_nav: float,
    ) -> str | None:
        """Return a readable reason if a tradable action has collapsed to zero."""
        assert action.ticker is not None
        execution_price = self._market.get_open_prices(decision_week + 1).get(action.ticker)
        if execution_price is None:
            return None

        resolved_shares = self._executor.resolve_shares(
            action=action,
            reference_price=float(execution_price),
            state=state,
            batch_start_nav=batch_start_nav,
        )
        if self._executor.is_effectively_zero_shares(resolved_shares):
            return self._default_zero_size_skip_reason(action)

        estimated_trade_value = abs(float(resolved_shares) * float(execution_price))
        if self._executor.is_effectively_zero_trade_value(estimated_trade_value):
            return self._default_zero_size_skip_reason(action)

        return None

    @staticmethod
    def _default_zero_size_skip_reason(action: Action) -> str:
        """Return a participant-friendly explanation for a zero-size skip."""
        if action.action_type == ActionType.BUY:
            return (
                "One planned buy became too small after the simulator applied its rules and execution-time pricing, so it was skipped."
            )
        if action.action_type in {ActionType.SELL, ActionType.REDUCE}:
            return (
                "One planned sale became too small or no longer had shares available at execution time, so it was skipped."
            )
        return (
            "One planned trade became too small to execute after the simulator applied its rules, so it was skipped."
        )

    @staticmethod
    def _execution_status_for_action(
        *,
        action: Action,
        validation_result: ValidationResult,
        execution_result: ExecutionResult | None,
        gap_clipped_to_zero: bool,
        execution_skip_reason: str | None,
    ) -> str:
        if validation_result.outcome == ValidationOutcome.REJECTED:
            return "rejected"
        if action.action_type in {ActionType.SET_STOP, ActionType.REMOVE_STOP, ActionType.HOLD}:
            return "not_applicable"
        if execution_result is not None:
            return "executed"
        if execution_skip_reason is not None:
            return "skipped_zero_size"
        if gap_clipped_to_zero:
            return "gap_clipped_to_zero"
        return "not_executed"

    @staticmethod
    def _resolve_ticker_universe(
        config: SimulatorConfig,
        market: MarketReplay,
    ) -> tuple[str, ...]:
        if config.ticker_universe:
            missing = sorted(set(config.ticker_universe) - set(market.available_tickers))
            if missing:
                raise ValueError(
                    "Configured ticker_universe contains tickers missing from market data: "
                    f"{missing}"
                )
            return tuple(config.ticker_universe)
        return tuple(market.available_tickers)
