"""Deterministic checks that gate the agent's final report."""

import math


REPAIRABLE_CHECKS = frozenset({"execution_ok", "no_lookahead"})
REQUIRED_CHECKS = frozenset({"execution_ok", "no_lookahead", "metrics_present", "metrics_valid", "equity_curve_valid", "metrics_match_equity"})


def _finite_number(value) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def evaluate_result(result: dict) -> list[dict]:
    executed = isinstance(result, dict) and result.get("ok") is True
    checks = [{
        "name": "execution_ok",
        "passed": executed,
        "detail": "Harness returned a result" if executed else str(result.get("error", "No valid harness result")) if isinstance(result, dict) else "Harness result was not a JSON object",
        "retryable": isinstance(result, dict) and result.get("retryable") is True,
    }]
    if not executed:
        return checks

    leakage = result.get("lookahead", {})
    leakage_detail = str(leakage.get("detail", "Lookahead result missing")) if isinstance(leakage, dict) else "Lookahead result is malformed"
    checks.append({
        "name": "no_lookahead",
        "passed": isinstance(leakage, dict) and leakage.get("passed") is True,
        "detail": leakage_detail,
    })
    metrics = result.get("metrics", {})
    required = ("total_return", "max_drawdown", "trades", "sharpe", "equity")
    present = isinstance(metrics, dict) and all(key in metrics for key in required)
    checks.append({
        "name": "metrics_present",
        "passed": present,
        "detail": "Harness returned all required metric fields" if present else "One or more required metric fields are missing",
    })
    equity = metrics.get("equity") if isinstance(metrics, dict) else None
    numeric = present and all(
        _finite_number(metrics[key])
        for key in ("total_return", "max_drawdown", "sharpe")
    )
    trades_valid = present and type(metrics["trades"]) is int and metrics["trades"] >= 0
    ranges_valid = (
        numeric
        and trades_valid
        and metrics["total_return"] > -1.0
        and -1.0 <= metrics["max_drawdown"] <= 0.0
        and (not isinstance(equity, (list, tuple)) or metrics["trades"] <= max(0, len(equity) - 1))
    )
    checks.append({
        "name": "metrics_valid",
        "passed": bool(numeric and trades_valid and ranges_valid),
        "detail": "Metrics are finite and within valid return, drawdown, and trade-count ranges" if numeric and trades_valid and ranges_valid else "Metrics contain invalid values or fall outside valid ranges",
    })
    equity_valid = (
        isinstance(equity, (list, tuple))
        and len(equity) >= 3
        and all(_finite_number(value) and value > 0 for value in equity)
        and math.isclose(equity[0], 1.0, rel_tol=0, abs_tol=1e-6)
    )
    checks.append({
        "name": "equity_curve_valid",
        "passed": equity_valid,
        "detail": "Equity curve is finite, positive, and starts at 1.0" if equity_valid else "Equity curve is missing or malformed",
    })
    if numeric and equity_valid:
        peak = equity[0]
        expected_drawdown = 0.0
        for value in equity:
            peak = max(peak, value)
            expected_drawdown = min(expected_drawdown, value / peak - 1.0)
        metrics_consistent = (
            math.isclose(metrics["total_return"], equity[-1] - 1.0, rel_tol=0, abs_tol=0.0001)
            and math.isclose(metrics["max_drawdown"], expected_drawdown, rel_tol=0, abs_tol=0.0001)
        )
    else:
        metrics_consistent = False
    checks.append({
        "name": "metrics_match_equity",
        "passed": metrics_consistent,
        "detail": "Reported return and drawdown match the equity curve" if metrics_consistent else "Reported return or drawdown does not match the equity curve",
    })
    return checks


def reportable(checks: list[dict]) -> bool:
    return (
        len(checks) == len(REQUIRED_CHECKS)
        and all(isinstance(check, dict) and check.get("passed") is True for check in checks)
        and {check.get("name") for check in checks} == REQUIRED_CHECKS
    )


def retry_eligible(checks: list[dict]) -> bool:
    """Only retry failures a repaired candidate can plausibly fix."""
    failures = [check for check in checks if not check["passed"]]
    return bool(failures) and all(
        check["name"] in REPAIRABLE_CHECKS
        and (check["name"] != "execution_ok" or check.get("retryable") is True)
        for check in failures
    )
