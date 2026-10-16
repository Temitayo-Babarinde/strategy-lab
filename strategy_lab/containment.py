"""A fixed, non-destructive sandbox probe. Never executed on the host."""
import json
import os

PROBE = '''import json, os, socket
names = ["DAYTONA_API_KEY", "ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY", "SPECTRON_API_KEY", "ANCHORBROWSER_API_KEY"]
present = [name for name in names if os.environ.get(name)]
connected = False
outcome = ""
try:
    with socket.create_connection(("1.1.1.1", 443), timeout=3):
        connected = True
        outcome = "Connection succeeded"
except OSError as error:
    outcome = type(error).__name__
print("PROBE_JSON:" + json.dumps({"secrets_absent": not present, "network_connected": connected, "network_outcome": outcome}))
'''


def run_containment_demo():
    if not os.getenv("DAYTONA_API_KEY"):
        return {"ok": False, "error": "Daytona is required; this probe never runs locally."}
    from daytona import CreateSandboxFromImageParams, Daytona, Resources
    sandbox = None
    result = {"ok": False}
    try:
        client = Daytona()
        sandbox = client.create(CreateSandboxFromImageParams(
            image="python:3.12-slim-bookworm", language="python",
            network_block_all=True, ephemeral=True, auto_stop_interval=0,
            resources=Resources(cpu=1, memory=2, disk=8)), timeout=60)
        sandbox.fs.upload_file(PROBE.encode(), "probe.py", timeout=15)
        response = sandbox.process.exec("python3 probe.py", timeout=15)
        lines = [line[11:] for line in response.result.splitlines() if line.startswith("PROBE_JSON:")]
        if response.exit_code != 0 or len(lines) != 1:
            raise ValueError("Probe did not return one successful evidence record")
        evidence = json.loads(lines[0])
        if not isinstance(evidence, dict) or type(evidence.get("secrets_absent")) is not bool or type(evidence.get("network_connected")) is not bool:
            raise ValueError("Invalid probe evidence")
        result = {"ok": True, **evidence, "network_policy": "network_block_all", "sandbox_id": sandbox.id}
    except Exception as error:
        result = {"ok": False, "error": f"Probe failed ({type(error).__name__})"}
    finally:
        if sandbox is not None:
            try:
                client.delete(sandbox, wait=True, timeout=30)
                result["cleaned_up"] = True
            except Exception:
                result.update(ok=False, cleaned_up=False, error="Sandbox cleanup failed")
    return result
