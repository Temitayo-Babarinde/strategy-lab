"""Run either trusted local demo functions or user code in a Daytona sandbox."""

from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path

from strategy_lab import harness
from strategy_lab.strategies import deliberately_leaky, momentum

ROOT = Path(__file__).resolve().parents[1]


def read_demo_csv() -> bytes:
    return (ROOT / "data" / "demo_prices.csv").read_bytes()


def parse_prices_csv(content: bytes) -> tuple[list[float], list[dict[str, str]]]:
    """Validate an uploaded close-price CSV and return prices plus preview rows."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV must use UTF-8 text encoding.") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if not reader.fieldnames or "close" not in {name.strip().lower() for name in reader.fieldnames}:
        raise ValueError("CSV needs a 'close' column. A 'timestamp' column is optional.")
    close_field = next(name for name in reader.fieldnames if name.strip().lower() == "close")
    rows: list[dict[str, str]] = []
    prices: list[float] = []
    for line_number, row in enumerate(reader, start=2):
        raw_close = row.get(close_field)
        try:
            price = float(raw_close) if raw_close is not None else math.nan
        except (TypeError, ValueError):
            raise ValueError(f"Row {line_number} has a non-numeric close price.") from None
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"Row {line_number} must have a finite, positive close price.")
        prices.append(price)
        rows.append({key: value or "" for key, value in row.items() if key is not None})
    if len(prices) < 3:
        raise ValueError("CSV needs at least three rows of prices.")
    return prices, rows


def read_demo_prices() -> list[float]:
    return parse_prices_csv(read_demo_csv())[0]


def run_local_demo(stage: str, prices: list[float] | None = None) -> dict:
    """Run only built-in trusted functions; never exec generated source locally."""
    if stage not in {"initial", "repaired"}:
        raise ValueError("Unknown local demo stage")
    prices = prices if prices is not None else read_demo_prices()
    fn = deliberately_leaky if stage == "initial" else momentum
    return harness.evaluate(prices, fn)


def run_daytona(strategy_source: str, prices_csv: bytes | None = None) -> dict:
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
        sandbox.fs.upload_file(prices_csv if prices_csv is not None else read_demo_csv(), "data.csv", timeout=15)
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
