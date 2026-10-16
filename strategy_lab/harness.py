"""Standard-library-only backtest harness, intended to run inside Daytona."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
import traceback
from typing import Callable


def load_prices(path: str = "data.csv") -> list[float]:
    with open(path, newline="", encoding="utf-8") as source:
        return [float(row["close"]) for row in csv.DictReader(source)]


def load_strategy(path: str = "strategy.py") -> Callable[[list[float]], list[int]]:
    spec = importlib.util.spec_from_file_location("user_strategy", path)
    if spec is None or spec.loader is None:
        raise ValueError("Could not load strategy file")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "signals", None)):
        raise ValueError("Strategy must define signals(prices)")
    return module.signals


def backtest(prices: list[float], signals: list[int]) -> dict:
    if len(prices) < 3:
        raise ValueError("Need at least three price rows")
    if any(not math.isfinite(price) or price <= 0 for price in prices):
        raise ValueError("prices must be finite positive numbers")
    if len(signals) != len(prices):
        raise ValueError("signals must return one value per price row")
    if any(type(signal) is not int or signal not in (-1, 0, 1) for signal in signals):
        raise ValueError("signals must contain only -1, 0, or 1")

    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    returns: list[float] = []
    position = 0
    trades = 0
    for index in range(1, len(prices)):
        # A signal observed at t-1 is applied to the next close-to-close return.
        next_position = int(signals[index - 1])
        if next_position != position:
            trades += 1
        cost = abs(next_position - position) * 0.0001
        bar_return = (prices[index] / prices[index - 1]) - 1.0
        strategy_return = next_position * bar_return - cost
        if strategy_return <= -1:
            raise ValueError("A single bar loss exhausted the simulated account")
        equity *= 1.0 + strategy_return
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1.0)
        returns.append(strategy_return)
        position = next_position

    mean = sum(returns) / len(returns)
    variance = sum((item - mean) ** 2 for item in returns) / max(1, len(returns) - 1)
    stdev = math.sqrt(variance)
    # The supplied series has no reliable bar-frequency contract, so keep this
    # as an unannualized per-bar ratio rather than implying a calendar cadence.
    sharpe = mean / stdev if stdev else 0.0
    return {
        "total_return": round(equity - 1.0, 6),
        "max_drawdown": round(max_drawdown, 6),
        "trades": trades,
        "sharpe": round(sharpe, 4),
        "equity": [round(value, 5) for value in _equity_curve(prices, signals)],
    }


def _equity_curve(prices: list[float], signals: list[int]) -> list[float]:
    curve = [1.0]
    position = 0
    for index in range(1, len(prices)):
        next_position = int(signals[index - 1])
        cost = abs(next_position - position) * 0.0001
        curve.append(curve[-1] * (1.0 + next_position * (prices[index] / prices[index - 1] - 1.0) - cost))
        position = next_position
    return curve


def _call_signals(prices: list[float], signal_fn: Callable[[list[float]], list[int]]) -> list[int]:
    supplied = list(prices)
    signals = list(signal_fn(supplied))
    if supplied != prices:
        raise ValueError("Strategy mutated the supplied prices")
    if len(signals) != len(prices) or any(type(value) is not int or value not in (-1, 0, 1) for value in signals):
        raise ValueError("signals must return one integer in {-1,0,1} per price row")
    return signals


def lookahead_check(prices: list[float], signal_fn: Callable[[list[float]], list[int]], expected_signals: list[int] | None = None) -> dict:
    full = _call_signals(prices, signal_fn)
    # Multiple deterministic prefixes make this failure repeatable in a live demo.
    cuts = sorted({max(2, len(prices) // 3), max(2, 2 * len(prices) // 3), len(prices) - 1})
    mismatches = int(expected_signals is not None and full != expected_signals)
    for cut in cuts:
        prefix_signals = _call_signals(prices[:cut], signal_fn)
        if prefix_signals != full[:cut]:
            mismatches += 1
    return {
        "passed": mismatches == 0,
        "mismatches": mismatches,
        "cuts_checked": len(cuts),
        "detail": "Signals stay stable when future rows are hidden" if mismatches == 0 else "Signals changed across repeat or prefix calls",
    }


def evaluate(prices: list[float], signal_fn: Callable[[list[float]], list[int]]) -> dict:
    prices = list(prices)
    signals = _call_signals(prices, signal_fn)
    metrics = backtest(prices, signals)
    return {
        "ok": True,
        "lookahead": lookahead_check(prices, signal_fn, signals),
        "metrics": metrics,
    }


def main() -> None:
    try:
        prices = load_prices(sys.argv[1] if len(sys.argv) > 1 else "data.csv")
        result = evaluate(prices, load_strategy(sys.argv[2] if len(sys.argv) > 2 else "strategy.py"))
    except Exception as exc:
        result = {"ok": False, "error": str(exc), "trace": traceback.format_exc(limit=2)}
    print("RESULT_JSON:" + json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
