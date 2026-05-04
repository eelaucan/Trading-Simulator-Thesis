"""Reusable backend runner for autonomous benchmark episodes."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from agents.benchmark_agent import AutonomousBenchmarkAgent, BenchmarkAgentConfig
from simulator.config import SimulatorConfig
from simulator.env import TradingEnvironment
from simulator.market import MarketReplay
from simulator.metrics import SimulationMetrics
from simulator.observation import Observation
from simulator.state import PortfolioState


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """Outputs from one completed autonomous benchmark episode."""

    env: TradingEnvironment
    agent: AutonomousBenchmarkAgent
    metrics: SimulationMetrics
    final_state: PortfolioState
    initial_observation: Observation
    output_paths: dict[str, Path]


def create_benchmark_agent_config(simulator_config: SimulatorConfig) -> BenchmarkAgentConfig:
    """Align agent sizing constraints with the active simulator rules."""
    cap_buffer = 0.005
    simulator_stock_cap = max(0.001, float(simulator_config.single_stock_cap) - cap_buffer)
    simulator_hhi_cap = max(0.001, math.sqrt(float(simulator_config.hhi_cap)) - cap_buffer)
    return BenchmarkAgentConfig(
        cash_reserve=max(0.05, float(simulator_config.cash_buffer)),
        max_position_weight=min(0.30, simulator_stock_cap, simulator_hhi_cap),
        max_turnover=float(simulator_config.turnover_cap),
        max_actions_per_step=int(simulator_config.max_actions_per_step),
    )


def run_agent_episode(
    env: TradingEnvironment,
    agent: AutonomousBenchmarkAgent,
) -> AgentRunResult:
    """Run ``agent`` through ``env`` until the episode terminates."""
    agent.reset_log()
    observation, state = env.reset()
    initial_observation = observation
    done = env.done

    while not done:
        actions = agent.decide(observation)
        observation, state, done, _info = env.step(actions)

    metrics = env.get_metrics()
    return AgentRunResult(
        env=env,
        agent=agent,
        metrics=metrics,
        final_state=state,
        initial_observation=initial_observation,
        output_paths={},
    )


def run_benchmark_agent(
    data_path: str | Path,
    *,
    simulator_config: SimulatorConfig | None = None,
    output_dir: str | Path | None = None,
    output_prefix: str = "benchmark_agent",
) -> AgentRunResult:
    """Build and run the default benchmark agent on a weekly OHLCV CSV."""
    market = MarketReplay(data_path)
    config = simulator_config or SimulatorConfig(
        initial_cash=100_000.0,
        ticker_universe=market.available_tickers,
    )
    env = TradingEnvironment(market=market, config=config)
    agent = AutonomousBenchmarkAgent(create_benchmark_agent_config(config))
    result = run_agent_episode(env, agent)
    if output_dir is None:
        return result

    output_paths = export_agent_run_outputs(
        result=result,
        output_dir=output_dir,
        output_prefix=output_prefix,
    )
    return AgentRunResult(
        env=result.env,
        agent=result.agent,
        metrics=result.metrics,
        final_state=result.final_state,
        initial_observation=result.initial_observation,
        output_paths=output_paths,
    )


def export_agent_run_outputs(
    *,
    result: AgentRunResult,
    output_dir: str | Path,
    output_prefix: str = "benchmark_agent",
) -> dict[str, Path]:
    """Export metrics, simulator logs, decision logs, and portfolio path."""
    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    paths = {
        "metrics": output_root / f"{output_prefix}_metrics.json",
        "action_log": output_root / f"{output_prefix}_action_log.csv",
        "batch_log": output_root / f"{output_prefix}_batch_log.csv",
        "batch_log_jsonl": output_root / f"{output_prefix}_batch_log.jsonl",
        "validation_log": output_root / f"{output_prefix}_validation_log.csv",
        "execution_log": output_root / f"{output_prefix}_execution_log.csv",
        "decision_log": output_root / f"{output_prefix}_decision_log.csv",
        "decision_log_jsonl": output_root / f"{output_prefix}_decision_log.jsonl",
        "weekly_returns": output_root / f"{output_prefix}_weekly_returns.csv",
        "equity_curve": output_root / f"{output_prefix}_equity_curve.csv",
        "final_holdings": output_root / f"{output_prefix}_final_holdings.csv",
    }

    _write_json(paths["metrics"], _metrics_to_json(result.metrics))
    result.env.logger.to_action_dataframe(include_internal=True).to_csv(
        paths["action_log"],
        index=False,
    )
    result.env.logger.to_batch_dataframe().to_csv(paths["batch_log"], index=False)
    result.env.logger.export_jsonl(paths["batch_log_jsonl"])
    result.metrics.validation_log_df.to_csv(paths["validation_log"], index=False)
    result.metrics.execution_log_df.to_csv(paths["execution_log"], index=False)
    result.agent.export_decision_csv(paths["decision_log"])
    result.agent.export_decision_jsonl(paths["decision_log_jsonl"])
    _weekly_returns_frame(result.metrics).to_csv(paths["weekly_returns"], index=False)
    _equity_curve_frame(result.final_state).to_csv(paths["equity_curve"], index=False)
    _final_holdings_frame(result.final_state).to_csv(paths["final_holdings"], index=False)
    return paths


def _metrics_to_json(metrics: SimulationMetrics) -> dict[str, Any]:
    return {
        "total_return": metrics.total_return,
        "max_drawdown": metrics.max_drawdown,
        "realized_vol": metrics.realized_vol,
        "avg_weekly_turnover": metrics.avg_weekly_turnover,
        "avg_hhi": metrics.avg_hhi,
        "max_hhi": metrics.max_hhi,
        "blow_up_flag": metrics.blow_up_flag,
        "sharpe_ratio": metrics.sharpe_ratio,
        "n_invalid_attempts": metrics.n_invalid_attempts,
        "n_clipped_trades": metrics.n_clipped_trades,
        "n_stop_triggers": metrics.n_stop_triggers,
        "n_gap_adjustments": metrics.n_gap_adjustments,
        "vol_rule_activation_week": metrics.vol_rule_activation_week,
    }


def _weekly_returns_frame(metrics: SimulationMetrics) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "week_index": list(metrics.weekly_returns.index),
            "weekly_return": metrics.weekly_returns.tolist(),
        }
    )


def _equity_curve_frame(state: PortfolioState) -> pd.DataFrame:
    nav_values = list(state.nav_history)
    start_week = int(state.week_index) - len(nav_values) + 1
    return pd.DataFrame(
        {
            "decision_step": list(range(1, len(nav_values) + 1)),
            "week_index": [start_week + offset for offset in range(len(nav_values))],
            "nav": nav_values,
        }
    )


def _final_holdings_frame(state: PortfolioState) -> pd.DataFrame:
    nav = max(float(state.total_nav), 1e-12)
    shares = state.shares_dict()
    market_values = state.market_value_dict()
    cost_basis = state.cost_basis_dict()
    rows = []
    for ticker in sorted(shares):
        rows.append(
            {
                "ticker": ticker,
                "shares": float(shares[ticker]),
                "market_value": float(market_values.get(ticker, 0.0)),
                "weight": float(market_values.get(ticker, 0.0)) / nav,
                "cost_basis": float(cost_basis.get(ticker, 0.0)),
            }
        )
    rows.append(
        {
            "ticker": "CASH",
            "shares": 0.0,
            "market_value": float(state.cash),
            "weight": float(state.cash) / nav,
            "cost_basis": 0.0,
        }
    )
    return pd.DataFrame(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
