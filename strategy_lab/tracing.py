"""Phoenix OSS tracing with optional Arize AX Cloud support."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse


def load_phoenix_environment() -> None:
    """Load the local Phoenix CLI config without overriding shell settings."""
    config_path = Path(__file__).resolve().parents[1] / ".env.phoenix"
    if not config_path.is_file():
        return
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if key.strip().startswith("PHOENIX_"):
            os.environ.setdefault(key.strip(), value)


def make_tracer():
    load_phoenix_environment()
    phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    if phoenix_endpoint:
        from phoenix.otel import register

        parsed = urlparse(phoenix_endpoint)
        if not parsed.hostname:
            raise ValueError("PHOENIX_COLLECTOR_ENDPOINT must be a complete collector URL")
        protocol = "grpc" if parsed.port == 4317 or parsed.scheme == "grpc" else "http/protobuf"
        if protocol == "http/protobuf" and not parsed.path.rstrip("/").endswith("/v1/traces"):
            parsed = parsed._replace(path=parsed.path.rstrip("/") + "/v1/traces")
        phoenix_endpoint = parsed.geturl()
        provider = register(
            endpoint=phoenix_endpoint,
            protocol=protocol,
            project_name=os.getenv("PHOENIX_PROJECT_NAME", "strategy-lab"),
            set_global_tracer_provider=False,
            batch=True,
            verbose=False,
        )
        return provider.get_tracer("strategy-lab"), provider

    if not (os.getenv("ARIZE_SPACE_ID") and os.getenv("ARIZE_API_KEY")):
        return None, None
    from arize.otel import register as register_ax

    provider = register_ax(
        space_id=os.environ["ARIZE_SPACE_ID"],
        api_key=os.environ["ARIZE_API_KEY"],
        project_name="trustworthy-backtest-agent",
        set_global_tracer_provider=False,
        verbose=False,
    )
    return provider.get_tracer("trustworthy-backtest-agent"), provider
