"""
main.py — Minimal runnable demonstration of the trading simulator.

Runs two strategies side-by-side:
  1. Pure HOLD (all-cash passive benchmark)
  2. Equal-weight buy-and-hold (invests 10% of NAV in each of 3 stocks)

Usage:
    python main.py
    python main.py --data data/sample/weekly_ohlcv_synthetic.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator import SimulatorConfig, TradingEnvironment, MarketReplay
from simulator.actions import Action, ActionType, QuantityType
from simulator.metrics import SimulationMetrics


# ------------------------------------------------------------------ #
# Strategy helpers                                                     #
# ------------------------------------------------------------------ #

def hold_strategy(_obs) -> list:
    """Always HOLD — passive cash benchmark."""
    return [Action(action_type=ActionType.HOLD)]


def buy_and_hold_strategy(obs, tickers, already_invested: bool) -> list:
    """
    At week 0: invest 10% of NAV in each of the first 3 tickers.
    Thereafter: hold.
    """
    if not already_invested:
        return [
            Action(
                action_type=ActionType.BUY,
                ticker=t,
                quantity=0.10,
                quantity_type=QuantityType.NAV_FRACTION,
            )
            for t in tickers[:3]
        ]
    return [Action(action_type=ActionType.HOLD)]


# ------------------------------------------------------------------ #
# Runner                                                               #
# ------------------------------------------------------------------ #

def run(data_path: str) -> None:
    print(f"\n{'='*60}")
    print("  TRADING SIMULATOR — DEMO RUN")
    print(f"{'='*60}\n")
    print(f"Data: {data_path}\n")

    # ---- Strategy 1: HOLD ----
    cfg = SimulatorConfig(initial_cash=100_000.0)
    market = MarketReplay(data_path)    
    env = TradingEnvironment(market, cfg)
    obs = env.reset()
    done = False

    while not done:
        actions = hold_strategy(obs)
        obs, state, done, _ = env.step(actions)

    m_hold = env.compute_metrics()
    _print_metrics("HOLD (all-cash)", m_hold)

    # ---- Strategy 2: Buy-and-hold ----
    cfg2 = SimulatorConfig(initial_cash=100_000.0)
    market2 = MarketReplay(data_path)
    env2 = TradingEnvironment(market2, cfg2)
    obs2 = env2.reset()
    done2 = False
    invested = False
    tickers = market2.ticker_universe

    while not done2:
        actions = buy_and_hold_strategy(obs2, tickers, invested)
        obs2, state2, done2, _ = env2.step(actions)
        invested = True

    m_bah = env2.compute_metrics()
    _print_metrics("Buy-and-Hold (10% each)", m_bah)

    # ---- Export logs ----
    log_dir = ROOT / "output"
    log_dir.mkdir(exist_ok=True)
    env.logger.export_csv(str(log_dir / "hold_audit_log.csv"))
    env2.logger.export_csv(str(log_dir / "bah_audit_log.csv"))
    env2.logger.export_jsonl(str(log_dir / "bah_audit_log.jsonl"))
    print(f"\nAudit logs exported to {log_dir}/")


def _print_metrics(label: str, m: SimulationMetrics) -> None:
    sharpe = f"{m.sharpe_ratio:.2f}" if m.sharpe_ratio is not None else "N/A"
    print(f"{'─'*40}")
    print(f"  Strategy : {label}")
    print(f"{'─'*40}")
    print(f"  Total Return      : {m.total_return:+.2%}")
    print(f"  Max Drawdown      : {m.max_drawdown:.2%}")
    print(f"  Realized Vol (ann): {m.realized_vol:.2%}")
    print(f"  Sharpe Ratio      : {sharpe}")
    print(f"  Avg HHI           : {m.avg_hhi:.4f}")
    print(f"  Max HHI           : {m.max_hhi:.4f}")
    print(f"  Avg Weekly T/O    : {m.avg_weekly_turnover:.4f}")
    print(f"  Blow-up Flag      : {m.blow_up_flag}")
    print(f"  Invalid Actions   : {m.n_invalid_attempts}")
    print(f"  Clipped Trades    : {m.n_clipped_trades}")
    print(f"  Stop Triggers     : {m.n_stop_triggers}")
    print()


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="Run trading simulator demo")
    parser.add_argument(
        "--data",
        type=str,
        default=str(ROOT / "data" / "sample" / "weekly_ohlcv_synthetic.csv"),
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Run `python data/generate_synthetic.py` first.")
        sys.exit(1)

    run(str(data_path))


if __name__ == "__main__":
    main()
