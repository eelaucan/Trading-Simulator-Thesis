"""Streamlit app for local human trading sessions."""

from __future__ import annotations

from datetime import datetime
import html
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - runtime-only dependency message
    raise SystemExit(
        "Streamlit is required to run the UI. Install it with `pip install streamlit`."
    ) from exc

from agents.benchmark_agent import AutonomousBenchmarkAgent
from agents.runner import AgentRunResult, run_benchmark_agent
from simulator.actions import Action, ActionType, QuantityType
from simulator.config import SimulatorConfig
from simulator.env import TradingEnvironment
from simulator.market import MarketReplay
from simulator.metrics import SimulationMetrics
from simulator.observation import Observation
from simulator.state import PortfolioState
from ui.components import (
    apply_ui_theme,
    build_trade_planner_props,
    render_coach_placeholder,
    render_financial_status_panel,
    render_final_summary,
    render_holdings_panel,
    render_market_panel,
    render_pending_liquidations_panel,
    render_portfolio_insight_panel,
    render_portfolio_pie_chart,
    render_risk_panel,
    render_section_header,
    render_session_bar,
    render_session_setup,
    render_step_feedback,
)
from ui.export import export_session_results
from ui.session import SessionMetadata, SessionStatus, condition_display_label
from ui_ts.python.trade_planner_component import (
    render_trade_planner_component,
    trade_planner_component_available,
)


_STATUS_KEY = "ui_session_status"
_METADATA_KEY = "ui_session_metadata"
_ENV_KEY = "ui_environment"
_OBS_KEY = "ui_observation"
_STATE_KEY = "ui_portfolio_state"
_ACTION_BATCH_KEY = "ui_action_batch"
_LAST_STEP_INFO_KEY = "ui_last_step_info"
_METRICS_KEY = "ui_metrics"
_EXPORT_DIR_KEY = "ui_export_dir"
_STEP_ERROR_KEY = "ui_step_error"
_PLANNER_EVENT_KEY = "ui_trade_planner_event_id"
_AGENT_KEY = "ui_autonomous_agent"
_RUN_MODE_KEY = "ui_run_mode"
_AI_RESULT_KEY = "ui_ai_result"
_AI_DECISION_LOG_KEY = "ui_ai_decision_log"
_AI_EXPORT_PATHS_KEY = "ui_ai_export_paths"


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="Thesis Trading Simulator",
        page_icon=":bar_chart:",
        layout="wide",
    )
    apply_ui_theme()
    _initialize_session_state()

    status = st.session_state[_STATUS_KEY]
    run_mode = st.session_state.get(_RUN_MODE_KEY, "human")
    if status == SessionStatus.NOT_STARTED:
        _render_setup_screen()
        return

    metadata = st.session_state[_METADATA_KEY]
    if metadata is None:
        _reset_ui_session()
        st.rerun()
        return

    _render_sidebar(metadata, status, run_mode)

    state = st.session_state[_STATE_KEY]
    env = st.session_state[_ENV_KEY]
    metrics = st.session_state[_METRICS_KEY]
    if state is None or env is None:
        _reset_ui_session()
        st.rerun()
        return

    if run_mode == "ai_benchmark":
        st.title("AI Benchmark Simulation")
        st.caption(
            "The autonomous benchmark completed the episode in the backend using the simulator environment."
        )
        if metrics is None:
            metrics = env.get_metrics()
            st.session_state[_METRICS_KEY] = metrics
        _render_ai_finished_screen(
            metadata=metadata,
            result=st.session_state.get(_AI_RESULT_KEY),
            env=env,
            state=state,
            metrics=metrics,
            decision_log=st.session_state.get(_AI_DECISION_LOG_KEY),
            export_paths=st.session_state.get(_AI_EXPORT_PATHS_KEY) or {},
        )
        return

    observation = st.session_state[_OBS_KEY]
    if observation is None:
        _reset_ui_session()
        st.rerun()
        return

    st.title("Trading Session")
    if status == SessionStatus.RUNNING:
        render_session_bar(metadata, observation)
    else:
        st.caption("Session complete. Review the summary below.")

    if st.session_state[_STEP_ERROR_KEY]:
        st.error(st.session_state[_STEP_ERROR_KEY])

    if status == SessionStatus.RUNNING:
        _render_running_screen(
            metadata,
            env,
            observation,
            state,
            last_step_info=st.session_state[_LAST_STEP_INFO_KEY],
        )
        return

    if metrics is None:
        metrics = env.get_metrics()
        st.session_state[_METRICS_KEY] = metrics

    _render_finished_screen(
        metadata,
        env,
        state,
        metrics,
        agent=st.session_state.get(_AGENT_KEY),
        last_step_info=st.session_state[_LAST_STEP_INFO_KEY],
    )

def _render_setup_screen() -> None:
    detected_datasets = _discover_datasets()
    default_dataset_path = _default_dataset_path()
    submitted = render_session_setup(
        default_dataset_path=default_dataset_path,
        detected_datasets=detected_datasets,
        default_episode_name="pilot_episode_01",
    )
    if submitted is None:
        return

    run_mode = str(submitted.get("run_mode", "human"))
    try:
        if run_mode == "ai_benchmark":
            _start_ai_session(submitted)
        else:
            _start_human_session(submitted)
    except Exception as exc:  # pragma: no cover - UI error path
        st.error(f"Unable to start the session: {exc}")
        return

    st.rerun()


def _start_human_session(submitted: dict[str, str]) -> None:
    """Create a normal interactive human session."""
    dataset_path = _resolve_dataset_path(submitted["dataset_path"])
    market = MarketReplay(dataset_path)
    config = SimulatorConfig(ticker_universe=market.available_tickers)
    env = TradingEnvironment(market=market, config=config)
    observation, state = env.reset()
    started_at = datetime.now().astimezone()
    metadata = SessionMetadata(
        participant_id=submitted["participant_id"],
        condition=submitted["condition"],
        episode_name=submitted["episode_name"],
        dataset_path=str(dataset_path),
        started_at=started_at,
        decision_start_week=env.initial_decision_week,
        visible_history_weeks_at_start=_visible_history_weeks(observation),
        notes=submitted["notes"] or None,
    )
    session_status = SessionStatus.FINISHED if env.done else SessionStatus.RUNNING
    metrics = None
    if session_status == SessionStatus.FINISHED:
        metadata = metadata.mark_finished(started_at)
        metrics = env.get_metrics()

    _store_common_session_state(
        run_mode="human",
        status=session_status,
        metadata=metadata,
        env=env,
        observation=observation,
        state=state,
        metrics=metrics,
    )
    st.session_state[_AI_RESULT_KEY] = None
    st.session_state[_AI_DECISION_LOG_KEY] = None
    st.session_state[_AI_EXPORT_PATHS_KEY] = {}
    st.session_state[_AGENT_KEY] = None


def _start_ai_session(submitted: dict[str, str]) -> None:
    """Run a completed autonomous benchmark episode in the backend."""
    dataset_path = _resolve_dataset_path(submitted["dataset_path"])
    started_at = datetime.now().astimezone()
    output_prefix = _ai_output_prefix(submitted, started_at)
    output_dir = PROJECT_ROOT / "output" / "ai_benchmark"

    with st.spinner("Running the AI benchmark episode..."):
        result = run_benchmark_agent(
            data_path=dataset_path,
            output_dir=output_dir,
            output_prefix=output_prefix,
        )

    metadata = SessionMetadata(
        participant_id=submitted["participant_id"],
        condition="ai_benchmark",
        episode_name=submitted["episode_name"],
        dataset_path=str(dataset_path),
        started_at=started_at,
        decision_start_week=result.env.initial_decision_week,
        visible_history_weeks_at_start=_visible_history_weeks(result.initial_observation),
        notes=submitted["notes"] or None,
    ).mark_finished(datetime.now().astimezone())
    decision_log = result.agent.to_decision_dataframe()

    _store_common_session_state(
        run_mode="ai_benchmark",
        status=SessionStatus.FINISHED,
        metadata=metadata,
        env=result.env,
        observation=None,
        state=result.final_state,
        metrics=result.metrics,
    )
    st.session_state[_AI_RESULT_KEY] = result
    st.session_state[_AI_DECISION_LOG_KEY] = decision_log
    st.session_state[_AI_EXPORT_PATHS_KEY] = result.output_paths
    st.session_state[_AGENT_KEY] = result.agent


def _store_common_session_state(
    *,
    run_mode: str,
    status: SessionStatus,
    metadata: SessionMetadata,
    env: TradingEnvironment,
    observation: Observation | None,
    state: PortfolioState,
    metrics: SimulationMetrics | None,
) -> None:
    """Store session state shared by human and AI startup paths."""
    st.session_state[_RUN_MODE_KEY] = run_mode
    st.session_state[_STATUS_KEY] = status
    st.session_state[_METADATA_KEY] = metadata
    st.session_state[_ENV_KEY] = env
    st.session_state[_OBS_KEY] = observation
    st.session_state[_STATE_KEY] = state
    st.session_state[_ACTION_BATCH_KEY] = []
    st.session_state[_LAST_STEP_INFO_KEY] = None
    st.session_state[_METRICS_KEY] = metrics
    st.session_state[_EXPORT_DIR_KEY] = None
    st.session_state[_STEP_ERROR_KEY] = None
    st.session_state[_PLANNER_EVENT_KEY] = None


def _render_sidebar(
    metadata: SessionMetadata,
    status: SessionStatus,
    run_mode: str,
) -> None:
    with st.sidebar:
        run_mode_label = _run_mode_label(run_mode)
        st.markdown(
            (
                "<div class='sidebar-card'>"
                "<p class='sidebar-eyebrow'>Session</p>"
                f"<div class='sidebar-status-pill'>{html.escape(status.value.replace('_', ' ').title())}</div>"
                "<div class='sidebar-row'><span class='sidebar-label'>Participant</span>"
                f"<span class='sidebar-value'>{html.escape(metadata.participant_id)}</span></div>"
                "<div class='sidebar-row'><span class='sidebar-label'>Session type</span>"
                f"<span class='sidebar-value'>{html.escape(condition_display_label(metadata.condition))}</span></div>"
                "<div class='sidebar-row'><span class='sidebar-label'>Run mode</span>"
                f"<span class='sidebar-value'>{html.escape(run_mode_label)}</span></div>"
                "<div class='sidebar-row'><span class='sidebar-label'>Episode</span>"
                f"<span class='sidebar-value'>{html.escape(metadata.episode_name)}</span></div>"
                "<div class='sidebar-row'><span class='sidebar-label'>Dataset</span>"
                f"<span class='sidebar-value'>{html.escape(Path(metadata.dataset_path).name)}</span></div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        if metadata.notes:
            st.caption(metadata.notes)

        export_dir = st.session_state[_EXPORT_DIR_KEY]
        if export_dir:
            st.success("Research files were written successfully.")
            st.caption(str(export_dir))

        if st.button(
            "Start a new local session",
            key="sidebar_reset_session",
            use_container_width=True,
        ):
            _reset_ui_session()
            st.rerun()


def _render_running_screen(
    metadata: SessionMetadata,
    env: TradingEnvironment,
    observation: Observation,
    state: PortfolioState,
    last_step_info: dict[str, object] | None,
) -> None:
    render_section_header("Market", "Current week visible data")
    render_market_panel(observation, key_prefix="market")

    st.divider()
    render_section_header("Portfolio", "Current positions, allocation, and performance")
    render_financial_status_panel(state)
    render_portfolio_insight_panel(state)

    st.markdown("**Holdings**")
    holdings_cols = st.columns([1.35, 1.0], gap="medium")
    with holdings_cols[0]:
        render_holdings_panel(state)
        render_portfolio_pie_chart(state)
    with holdings_cols[1]:
        render_risk_panel(state)
        render_pending_liquidations_panel(observation.pending_liquidations)

    st.divider()
    render_section_header("Trade Planner", "Build and submit your weekly decisions")

    current_batch = list(st.session_state[_ACTION_BATCH_KEY])
    planner_props = build_trade_planner_props(
        config=env.config,
        observation=observation,
        current_batch=current_batch,
    )
    planner_event: dict[str, object] | None = None

    if trade_planner_component_available():
        planner_event = render_trade_planner_component(
            props=planner_props,
            key="trade_planner_component",
        )
    else:
        st.warning(
            "The custom trade planner bundle is not available locally yet. "
            "Build or restore `ui_ts/frontend/dist` to use the TypeScript planner."
        )

    if metadata.condition == "human_with_coach_placeholder":
        st.markdown("")
        render_coach_placeholder(metadata.condition)

    if _handle_trade_planner_event(env, planner_event):
        st.rerun()

    if last_step_info:
        st.divider()
        render_step_feedback(last_step_info)


def _render_finished_screen(
    metadata: SessionMetadata,
    env: TradingEnvironment,
    state: PortfolioState,
    metrics: SimulationMetrics,
    agent: AutonomousBenchmarkAgent | None,
    last_step_info: dict[str, object] | None,
) -> None:
    if last_step_info:
        render_step_feedback(last_step_info)
        st.divider()
    render_final_summary(
        metadata=metadata,
        state=state,
        metrics=metrics,
        export_path=st.session_state[_EXPORT_DIR_KEY],
    )
    st.divider()
    render_section_header(
        "Final portfolio review",
        "Equity path, ending allocation, and holdings available for comparison.",
    )
    render_portfolio_insight_panel(state)
    holdings_finish_cols = st.columns([1.4, 0.8], gap="medium")
    with holdings_finish_cols[0]:
        render_holdings_panel(state)
    with holdings_finish_cols[1]:
        render_portfolio_pie_chart(state)
    _render_export_controls(metadata, env, metrics, agent)
    with st.expander("Detailed research logs (optional)", expanded=False):
        tab_names = ["Action Log", "Batch Log", "Validation Log", "Execution Log"]
        if agent is not None:
            tab_names.append("AI Decision Log")
        log_tabs = st.tabs(tab_names)
        with log_tabs[0]:
            st.dataframe(env.logger.to_action_dataframe(include_internal=True), use_container_width=True)
        with log_tabs[1]:
            st.dataframe(env.logger.to_batch_dataframe(), use_container_width=True)
        with log_tabs[2]:
            st.dataframe(metrics.validation_log_df, use_container_width=True)
        with log_tabs[3]:
            st.dataframe(metrics.execution_log_df, use_container_width=True)
        if agent is not None:
            with log_tabs[4]:
                st.dataframe(agent.to_decision_dataframe(), use_container_width=True)


def _render_ai_finished_screen(
    *,
    metadata: SessionMetadata,
    result: AgentRunResult | None,
    env: TradingEnvironment,
    state: PortfolioState,
    metrics: SimulationMetrics,
    decision_log: object | None,
    export_paths: dict[str, Path],
) -> None:
    """Render completed autonomous benchmark outputs without the human planner."""
    agent = result.agent if result is not None else st.session_state.get(_AGENT_KEY)
    if decision_log is None and agent is not None:
        decision_log = agent.to_decision_dataframe()

    render_final_summary(
        metadata=metadata,
        state=state,
        metrics=metrics,
        export_path=st.session_state[_EXPORT_DIR_KEY],
    )

    st.divider()
    render_section_header(
        "AI portfolio review",
        "Completed autonomous trajectory, ending allocation, and final holdings.",
    )
    render_portfolio_insight_panel(state)
    ai_holdings_cols = st.columns([1.4, 0.8], gap="medium")
    with ai_holdings_cols[0]:
        render_holdings_panel(state)
    with ai_holdings_cols[1]:
        render_portfolio_pie_chart(state)

    if export_paths:
        st.markdown("**Benchmark output files**")
        st.dataframe(
            [
                {"Output": label, "Path": str(path)}
                for label, path in sorted(export_paths.items())
            ],
            use_container_width=True,
            hide_index=True,
        )

    _render_export_controls(metadata, env, metrics, agent)

    with st.expander("Detailed AI and simulator logs", expanded=True):
        log_tabs = st.tabs(
            ["AI Decision Log", "Action Log", "Batch Log", "Validation Log", "Execution Log"]
        )
        with log_tabs[0]:
            if decision_log is not None:
                st.dataframe(decision_log, use_container_width=True)
            else:
                st.info("No AI decision log is available for this run.")
        with log_tabs[1]:
            st.dataframe(env.logger.to_action_dataframe(include_internal=True), use_container_width=True)
        with log_tabs[2]:
            st.dataframe(env.logger.to_batch_dataframe(), use_container_width=True)
        with log_tabs[3]:
            st.dataframe(metrics.validation_log_df, use_container_width=True)
        with log_tabs[4]:
            st.dataframe(metrics.execution_log_df, use_container_width=True)


def _render_export_controls(
    metadata: SessionMetadata,
    env: TradingEnvironment,
    metrics: SimulationMetrics,
    agent: AutonomousBenchmarkAgent | None,
) -> None:
    st.subheader("Save research files")
    st.caption(
        "These files include your session metadata, simulator logs, and final metrics for later analysis."
    )
    if st.button("Write session files", type="primary", key="export_session_button"):
        export_dir = export_session_results(
            metadata=metadata,
            status=SessionStatus.FINISHED,
            env=env,
            metrics=metrics,
            agent_decision_log_df=(
                agent.to_decision_dataframe() if agent is not None else None
            ),
            agent_decision_records=agent.decision_records if agent is not None else None,
            output_root=PROJECT_ROOT / "output" / "sessions",
        )
        st.session_state[_EXPORT_DIR_KEY] = str(export_dir)
        st.success(f"Session files written to {export_dir}")


def _submit_batch() -> bool:
    env = st.session_state[_ENV_KEY]
    metadata = st.session_state[_METADATA_KEY]
    current_batch = list(st.session_state[_ACTION_BATCH_KEY])
    previous_state = st.session_state[_STATE_KEY]

    try:
        observation, state, done, info = env.step(current_batch)
    except Exception as exc:  # pragma: no cover - UI error path
        st.session_state[_STEP_ERROR_KEY] = f"Step failed: {exc}"
        return False

    st.session_state[_STEP_ERROR_KEY] = None
    st.session_state[_OBS_KEY] = observation
    st.session_state[_STATE_KEY] = state
    st.session_state[_LAST_STEP_INFO_KEY] = _augment_step_info(
        previous_state=previous_state,
        next_state=state,
        info=info,
    )
    st.session_state[_ACTION_BATCH_KEY] = []

    if done:
        finished_metadata = metadata.mark_finished(datetime.now().astimezone())
        st.session_state[_METADATA_KEY] = finished_metadata
        st.session_state[_STATUS_KEY] = SessionStatus.FINISHED
        st.session_state[_METRICS_KEY] = env.get_metrics()
    return True


def _append_action_to_batch(
    current_batch: list[Action],
    action: Action,
    max_actions_per_step: int,
) -> list[Action]:
    cleaned_batch = [item for item in current_batch if item.action_type != ActionType.HOLD]
    if len(cleaned_batch) >= max_actions_per_step:
        raise ValueError("The current batch already contains the maximum number of actions.")
    if action.action_type == ActionType.HOLD and cleaned_batch:
        raise ValueError("A do-nothing week must be submitted on its own.")
    updated_batch = list(cleaned_batch)
    updated_batch.append(action)
    return updated_batch


def _handle_trade_planner_event(
    env: TradingEnvironment,
    planner_event: dict[str, object] | None,
) -> bool:
    """Process the latest event emitted by the TypeScript trade planner."""
    if not planner_event:
        return False

    event_id = str(planner_event.get("event_id", "")).strip()
    if not event_id or event_id == st.session_state.get(_PLANNER_EVENT_KEY):
        return False

    st.session_state[_PLANNER_EVENT_KEY] = event_id

    try:
        next_batch = _action_batch_from_component_payload(
            planner_event.get("actions"),
            env.config.max_actions_per_step,
        )
    except ValueError as exc:
        st.session_state[_STEP_ERROR_KEY] = f"Planner error: {exc}"
        return True

    st.session_state[_ACTION_BATCH_KEY] = next_batch
    st.session_state[_STEP_ERROR_KEY] = None

    event_type = str(planner_event.get("event_type", "plan_change")).strip().lower()
    if event_type == "submit":
        _submit_batch()
        return True
    return True


def _action_batch_from_component_payload(
    payload: object,
    max_actions_per_step: int,
) -> list[Action]:
    """Map the component payload back into validated Python Action objects."""
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError("Action payload must be a list of planned actions.")

    batch: list[Action] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each planned action must be an object.")
        action = _action_from_component_payload(item)
        batch = _append_action_to_batch(batch, action, max_actions_per_step)

    return batch


def _action_from_component_payload(payload: dict[str, object]) -> Action:
    """Create one Python Action from the custom component payload."""
    action_type_raw = payload.get("action_type")
    if not isinstance(action_type_raw, str):
        raise ValueError("Each planned action must include an action_type.")
    action_type = ActionType(action_type_raw)

    ticker_value = payload.get("ticker")
    ticker = ticker_value.strip() if isinstance(ticker_value, str) and ticker_value.strip() else None

    quantity_type_raw = payload.get("quantity_type")
    quantity_type = (
        None
        if quantity_type_raw in (None, "")
        else QuantityType(str(quantity_type_raw))
    )

    quantity_raw = payload.get("quantity")
    quantity = None if quantity_raw in (None, "") else float(quantity_raw)

    fraction_raw = payload.get("fraction")
    fraction = None if fraction_raw in (None, "") else float(fraction_raw)

    stop_price_raw = payload.get("stop_price")
    stop_price = None if stop_price_raw in (None, "") else float(stop_price_raw)

    return Action(
        action_type=action_type,
        ticker=ticker,
        quantity=quantity,
        quantity_type=quantity_type,
        fraction=fraction,
        stop_price=stop_price,
    )


def _augment_step_info(
    *,
    previous_state: PortfolioState,
    next_state: PortfolioState,
    info: dict[str, object],
) -> dict[str, object]:
    previous_shares = previous_state.shares_dict()
    next_shares = next_state.shares_dict()
    position_change_items: list[str] = []

    for ticker in sorted(set(previous_shares) | set(next_shares)):
        before_shares = float(previous_shares.get(ticker, 0.0))
        after_shares = float(next_shares.get(ticker, 0.0))
        if before_shares <= 1e-12 and after_shares > 1e-12:
            position_change_items.append(
                f"{ticker}: opened a new position and now hold {_format_share_count(after_shares)}."
            )
        elif before_shares > 1e-12 and after_shares <= 1e-12:
            position_change_items.append(f"{ticker}: fully removed from the portfolio.")
        elif after_shares > before_shares + 1e-12:
            position_change_items.append(
                f"{ticker}: increased from {_format_share_count(before_shares)} to {_format_share_count(after_shares)}."
            )
        elif after_shares < before_shares - 1e-12:
            position_change_items.append(
                f"{ticker}: reduced from {_format_share_count(before_shares)} to {_format_share_count(after_shares)}."
            )

    augmented = dict(info)
    augmented.update(
        {
            "cash_before": float(previous_state.cash),
            "cash_after": float(next_state.cash),
            "invested_before": float(max(0.0, previous_state.total_nav - previous_state.cash)),
            "invested_after": float(max(0.0, next_state.total_nav - next_state.cash)),
            "total_nav_before": float(previous_state.total_nav),
            "total_nav_after": float(next_state.total_nav),
            "holdings_before_count": sum(1 for shares in previous_shares.values() if shares > 1e-12),
            "holdings_after_count": sum(1 for shares in next_shares.values() if shares > 1e-12),
            "position_change_items": tuple(position_change_items),
        }
    )
    return augmented


def _format_share_count(value: float) -> str:
    rounded = round(float(value), 4)
    if rounded.is_integer():
        unit = "share" if int(rounded) == 1 else "shares"
        return f"{int(rounded):,} {unit}"
    return f"{rounded:,.4f} shares"


def _initialize_session_state() -> None:
    defaults = {
        _STATUS_KEY: SessionStatus.NOT_STARTED,
        _RUN_MODE_KEY: "human",
        _METADATA_KEY: None,
        _ENV_KEY: None,
        _OBS_KEY: None,
        _STATE_KEY: None,
        _ACTION_BATCH_KEY: [],
        _LAST_STEP_INFO_KEY: None,
        _METRICS_KEY: None,
        _EXPORT_DIR_KEY: None,
        _STEP_ERROR_KEY: None,
        _PLANNER_EVENT_KEY: None,
        _AGENT_KEY: None,
        _AI_RESULT_KEY: None,
        _AI_DECISION_LOG_KEY: None,
        _AI_EXPORT_PATHS_KEY: {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _reset_ui_session() -> None:
    for key in (
        _STATUS_KEY,
        _RUN_MODE_KEY,
        _METADATA_KEY,
        _ENV_KEY,
        _OBS_KEY,
        _STATE_KEY,
        _ACTION_BATCH_KEY,
        _LAST_STEP_INFO_KEY,
        _METRICS_KEY,
        _EXPORT_DIR_KEY,
        _STEP_ERROR_KEY,
        _PLANNER_EVENT_KEY,
        _AGENT_KEY,
        _AI_RESULT_KEY,
        _AI_DECISION_LOG_KEY,
        _AI_EXPORT_PATHS_KEY,
    ):
        if key in st.session_state:
            del st.session_state[key]
    _initialize_session_state()


def _default_dataset_path() -> str:
    return str(PROJECT_ROOT / "data" / "sample" / "weekly_ohlcv_synthetic.csv")


def _discover_datasets() -> list[str]:
    dataset_root = PROJECT_ROOT / "data"
    return sorted(str(path) for path in dataset_root.rglob("*.csv"))


def _resolve_dataset_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {candidate}")
    return candidate


def _visible_history_weeks(observation: Observation) -> int:
    history = observation.price_history
    if "_week_idx" in history.columns:
        return int(history["_week_idx"].nunique())
    return int(history["date"].nunique())


def _run_mode_label(run_mode: str) -> str:
    if run_mode == "ai_benchmark":
        return "AI Benchmark"
    return "Human"


def _ai_output_prefix(submitted: dict[str, str], started_at: datetime) -> str:
    participant = _safe_file_token(submitted.get("participant_id", "benchmark_agent"))
    episode = _safe_file_token(submitted.get("episode_name", "episode"))
    timestamp = started_at.strftime("%Y%m%dT%H%M%S")
    return f"{participant}_{episode}_{timestamp}"


def _safe_file_token(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value.strip())
    return cleaned or "benchmark_agent"


if __name__ == "__main__":
    main()
