"""
Trading Simulator — research-grade historical replay engine.

Primary entry point:

    from simulator import TradingEnvironment, SimulatorConfig, MarketReplay
    from simulator.actions import Action, ActionType, QuantityType

    config = SimulatorConfig(initial_cash=100_000)
    market = MarketReplay("data/sample/weekly_ohlcv_synthetic.csv", config.ticker_universe)
    env    = TradingEnvironment(market, config)
    obs    = env.reset()
"""
from .config import SimulatorConfig
from .env import TradingEnvironment
from .market import MarketReplay

__all__ = ["SimulatorConfig", "TradingEnvironment", "MarketReplay"]
