"""Run either trusted local demo functions or user code in a Daytona sandbox."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from strategy_lab import harness
from strategy_lab.strategies import deliberately_leaky, momentum

ROOT = Path(__file__).resolve().parents[1]


def read_demo_prices() -> list[float]:
    with (ROOT / "data" / "demo_prices.csv").open(newline="", encoding="utf-8") as source:
        return [float(row["close"]) for row in csv.DictReader(source)]


def run_local_demo(stage: str) -> dict:
    """Run only built-in trusted functions; never exec generated source locally."""
    if stage not in {"initial", "repaired"}:
        raise ValueError("Unknown local demo stage")
    prices = read_demo_prices()
    fn = deliberately_leaky if stage == "initial" else momentum
    return harness.evaluate(prices, fn)


def run_daytona(strategy_source: str) -> dict:
    """Upload the harness and candidate to a bounded, network-isolated sandbox."""
    if not os.getenv("DAYTONA_API_KEY"):
        raise RuntimeError("DAYTONA_API_KEY is not configured")
    try:
        from daytona import CreateSandboxFromImageParams, Daytona, Resources
    except Exception as error:
        return {"ok": False, "error": f"Daytona SDK could not load: {type(error).__name__}: {error}"}

    try:
        client = Daytona()
    except Exception as error:
        return {"ok": False, "error": f"Daytona client could not start: {type(error).__name__}: {error}"}
    sandbox = None
    result = None
    run_error = None
    cleanup_error = None
    try:
        sandbox = client.create(
            CreateSandboxFromImageParams(
                image="python:3.12-slim-bookworm",
                language="python",
                network_block_all=True,
                ephemeral=True,
                auto_stop_interval=0,
                resources=Resources(cpu=1, memory=2, disk=8),
            ),
            timeout=60,
        )
        sandbox.fs.upload_file((ROOT / "data" / "demo_prices.csv").read_bytes(), "data.csv", timeout=15)
        sandbox.fs.upload_file((ROOT / "strategy_lab" / "harness.py").read_bytes(), "harness.py", timeout=15)
        sandbox.fs.upload_file(strategy_source.encode("utf-8"), "strategy.py", timeout=15)
        response = sandbox.process.exec("python3 harness.py data.csv strategy.py", timeout=45)
        output = response.result if isinstance(response.result, str) else ""
        lines = [item for item in output.splitlines() if item.startswith("RESULT_JSON:")]
        if response.exit_code != 0:
            run_error = f"Sandbox process exited unsuccessfully (exit code {response.exit_code})"
        elif len(lines) != 1:
            run_error = "Sandbox must return exactly one result JSON record"
        else:
            result = json.loads(lines[0][len("RESULT_JSON:"):])
            if not isinstance(result, dict):
                run_error = "Sandbox result JSON was not an object"
            elif result.get("ok") is False:
                result["retryable"] = True
    except Exception as error:
        run_error = f"Daytona run failed: {type(error).__name__}: {error}"
    finally:
        if sandbox is not None:
            try:
                client.delete(sandbox, wait=True, timeout=30)
            except Exception as error:
                cleanup_error = f"Sandbox cleanup failed: {type(error).__name__}: {error}"

    if cleanup_error:
        return {"ok": False, "error": cleanup_error, "retryable": False}
    if run_error:
        return {"ok": False, "error": run_error, "retryable": False}
    if result is None:
        return {"ok": False, "error": "Sandbox returned no usable result", "retryable": False}
    return result
