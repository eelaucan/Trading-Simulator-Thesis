from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import pandas as pd

from agents.benchmark_agent import AutonomousBenchmarkAgent, BenchmarkAgentConfig
from agents.runner import create_benchmark_agent_config, run_agent_episode
from simulator.actions import ActionType, QuantityType
from simulator.config import SimulatorConfig
from simulator.env import TradingEnvironment
from simulator.market import MarketReplay
from simulator.observation import Observation
from simulator.state import PortfolioState


def _market_from_price_paths(tmp_path, paths: dict[str, list[float]]) -> MarketReplay:
    rows = []
    start = datetime(2020, 1, 6)
    for week in range(len(next(iter(paths.values())))):
        date = start + timedelta(weeks=week)
        for ticker, closes in paths.items():
            close = float(closes[week])
            previous = float(closes[week - 1]) if week > 0 else close
            open_price = previous
            high = max(open_price, close) * 1.01
            low = min(open_price, close) * 0.99
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": open_price,
                    "high": high,
                    "low": max(0.01, low),
                    "close": close,
                    "volume": 1_000_000,
                }
            )
    path = tmp_path / "weekly.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return MarketReplay(path)


def _observation_for_paths(
    tmp_path,
    paths: dict[str, list[float]],
    *,
    initial_decision_week: int = 12,
) -> Observation:
    market = _market_from_price_paths(tmp_path, paths)
    config = SimulatorConfig(
        ticker_universe=market.available_tickers,
        observation_history_weeks=initial_decision_week + 1,
        initial_decision_week=initial_decision_week,
        single_stock_cap=0.30,
        hhi_cap=0.40,
        turnover_cap=0.50,
    )
    env = TradingEnvironment(market=market, config=config)
    observation, _state = env.reset()
    return observation


def _replace_state(observation: Observation, state: PortfolioState) -> Observation:
    return replace(observation, portfolio_state=state)


def _state_with_holdings(
    observation: Observation,
    *,
    cash: float,
    holdings: dict[str, tuple[float, float]],
    total_nav: float = 100_000.0,
) -> PortfolioState:
    shares = {ticker: values[0] for ticker, values in holdings.items()}
    market_values = {ticker: values[1] for ticker, values in holdings.items()}
    cost_basis = {
        ticker: market_values[ticker] / shares[ticker]
        for ticker in shares
        if shares[ticker] > 0
    }
    return PortfolioState(
        week_index=observation.week_index,
        date=observation.date,
        cash=cash,
        shares=PortfolioState._to_tuple(shares),
        market_value=PortfolioState._to_tuple(market_values),
        total_nav=total_nav,
        realized_pnl=0.0,
        unrealized_pnl=PortfolioState._to_tuple({ticker: 0.0 for ticker in shares}),
        cost_basis=PortfolioState._to_tuple(cost_basis),
        stop_levels=tuple(),
        weekly_turnover=0.0,
        concentration_hhi=0.0,
        portfolio_volatility=None,
        nav_history=(total_nav,),
    )


def _default_paths() -> dict[str, list[float]]:
    return {
        "AAA": [100, 101, 102, 103, 104, 106, 108, 110, 112, 114, 117, 120, 124, 128, 132],
        "BBB": [100, 99, 98, 97, 96, 95, 93, 91, 90, 88, 87, 85, 84, 83, 82],
        "CCC": [100, 100, 101, 101, 102, 102, 103, 104, 105, 105, 106, 107, 108, 109, 110],
    }


def test_signal_generation_ranks_positive_momentum(tmp_path):
    observation = _observation_for_paths(tmp_path, _default_paths())
    agent = AutonomousBenchmarkAgent(
        BenchmarkAgentConfig(volatility_penalty=0.0, max_turnover=0.50)
    )

    signals = {signal.ticker: signal for signal in agent.compute_signals(observation)}

    assert signals["AAA"].score is not None and signals["AAA"].score > 0
    assert signals["BBB"].score is not None and signals["BBB"].score < 0
    assert signals["AAA"].score > signals["CCC"].score
    assert signals["AAA"].reason == "positive_momentum"


def test_position_selection_and_buy_action_construction(tmp_path):
    observation = _observation_for_paths(tmp_path, _default_paths())
    agent = AutonomousBenchmarkAgent(
        BenchmarkAgentConfig(
            volatility_penalty=0.0,
            max_positions=2,
            max_position_weight=0.30,
            max_turnover=0.50,
        )
    )

    actions = agent.decide(observation)

    buy_actions = [action for action in actions if action.action_type == ActionType.BUY]
    assert [action.ticker for action in buy_actions] == ["AAA", "CCC"]
    assert all(action.quantity_type == QuantityType.NAV_FRACTION for action in buy_actions)
    assert all(action.quantity is not None and action.quantity > 0 for action in buy_actions)


def test_no_positive_signals_returns_hold(tmp_path):
    paths = {
        "AAA": [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87],
        "BBB": [100, 99, 97, 96, 95, 93, 92, 91, 89, 88, 87, 86, 85, 84],
    }
    observation = _observation_for_paths(tmp_path, paths)
    agent = AutonomousBenchmarkAgent(BenchmarkAgentConfig(volatility_penalty=0.0))

    actions = agent.decide(observation)

    assert len(actions) == 1
    assert actions[0].action_type == ActionType.HOLD
    decision_log = agent.to_decision_dataframe()
    assert "hold_no_qualified_asset" in decision_log.iloc[0]["reasons"]


def test_empty_history_signals_are_ineligible():
    observation = type(
        "MinimalObservation",
        (),
        {
            "week_index": 0,
            "available_tickers": ["AAA"],
            "price_history": pd.DataFrame(),
        },
    )()
    agent = AutonomousBenchmarkAgent()

    signals = agent.compute_signals(observation)

    assert signals[0].ticker == "AAA"
    assert signals[0].eligible is False
    assert signals[0].reason == "insufficient_history"


def test_insufficient_cash_blocks_new_buy(tmp_path):
    observation = _observation_for_paths(tmp_path, _default_paths())
    low_cash_state = _state_with_holdings(observation, cash=4_000.0, holdings={})
    observation = _replace_state(observation, low_cash_state)
    agent = AutonomousBenchmarkAgent(
        BenchmarkAgentConfig(
            volatility_penalty=0.0,
            cash_reserve=0.05,
            max_position_weight=0.30,
            max_turnover=0.50,
        )
    )

    actions = agent.decide(observation)

    assert len(actions) == 1
    assert actions[0].action_type == ActionType.HOLD
    decision_log = agent.to_decision_dataframe()
    aaa_row = decision_log[decision_log["ticker"] == "AAA"].iloc[0]
    assert "insufficient_cash" in aaa_row["reasons"]


def test_already_at_max_positions_does_not_add_lower_ranked_name(tmp_path):
    observation = _observation_for_paths(tmp_path, _default_paths())
    state = _state_with_holdings(
        observation,
        cash=60_000.0,
        holdings={
            "AAA": (200.0, 20_000.0),
            "CCC": (200.0, 20_000.0),
        },
    )
    observation = _replace_state(observation, state)
    agent = AutonomousBenchmarkAgent(
        BenchmarkAgentConfig(
            volatility_penalty=0.0,
            max_positions=2,
            max_position_weight=0.20,
            max_turnover=0.50,
        )
    )

    actions = agent.decide(observation)

    assert len(actions) == 1
    assert actions[0].action_type == ActionType.HOLD
    assert "BBB" not in agent.decision_records[-1]["selected_tickers"]


def test_selling_existing_holding_with_negative_signal(tmp_path):
    observation = _observation_for_paths(tmp_path, _default_paths())
    state = _state_with_holdings(
        observation,
        cash=80_000.0,
        holdings={"BBB": (200.0, 20_000.0)},
    )
    observation = _replace_state(observation, state)
    agent = AutonomousBenchmarkAgent(
        BenchmarkAgentConfig(volatility_penalty=0.0, max_turnover=0.50)
    )

    actions = agent.decide(observation)

    assert any(
        action.action_type == ActionType.SELL
        and action.ticker == "BBB"
        and action.quantity_type == QuantityType.CLOSE_ALL
        for action in actions
    )


def test_decision_uses_only_visible_observation_not_future_rows(tmp_path):
    base_paths = _default_paths()
    changed_future_paths = {ticker: list(values) for ticker, values in base_paths.items()}
    changed_future_paths["BBB"][13] = 500.0
    changed_future_paths["BBB"][14] = 700.0

    observation_a = _observation_for_paths(tmp_path / "a", base_paths)
    observation_b = _observation_for_paths(tmp_path / "b", changed_future_paths)
    agent_a = AutonomousBenchmarkAgent(BenchmarkAgentConfig(volatility_penalty=0.0))
    agent_b = AutonomousBenchmarkAgent(BenchmarkAgentConfig(volatility_penalty=0.0))

    actions_a = agent_a.decide(observation_a)
    actions_b = agent_b.decide(observation_b)

    assert [
        (action.action_type, action.ticker, action.quantity, action.quantity_type)
        for action in actions_a
    ] == [
        (action.action_type, action.ticker, action.quantity, action.quantity_type)
        for action in actions_b
    ]


def test_agent_completes_full_environment_episode(tmp_path):
    market = _market_from_price_paths(tmp_path, _default_paths())
    config = SimulatorConfig(
        ticker_universe=market.available_tickers,
        observation_history_weeks=13,
        initial_decision_week=12,
        single_stock_cap=0.30,
        hhi_cap=0.40,
        turnover_cap=0.50,
    )
    env = TradingEnvironment(market=market, config=config)
    agent = AutonomousBenchmarkAgent(create_benchmark_agent_config(config))

    result = run_agent_episode(env, agent)

    assert result.env.done
    assert result.metrics.action_log_df is not None
    assert not result.agent.to_decision_dataframe().empty
