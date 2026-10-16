"""Offline regression and real-SDK contract checks. No live keys or network required."""

import copy
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx

from strategy_lab import agent, evals, harness, runner, sponsor_integrations as sponsors, tracing
from strategy_lab.strategies import LEAKY_SOURCE, MOMENTUM_SOURCE, deliberately_leaky, momentum


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"AWS_EC2_METADATA_DISABLED": "true"}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_seed_and_repair_gate(self):
        first = evals.evaluate_result(runner.run_local_demo("initial"))
        final = evals.evaluate_result(runner.run_local_demo("repaired"))
        self.assertFalse(evals.reportable(first))
        self.assertTrue(evals.retry_eligible(first))
        self.assertTrue(evals.reportable(final))
        self.assertFalse(evals.retry_eligible(final))

    def test_source_identity_matches_local_functions(self):
        prices = runner.read_demo_prices()
        for source, function in [(LEAKY_SOURCE, deliberately_leaky), (MOMENTUM_SOURCE, momentum)]:
            namespace = {}
            # These two sources are trusted repository fixtures, never model output.
            exec(source, namespace)
            self.assertEqual(namespace["signals"](prices), function(prices))

    def test_empty_or_partial_checks_never_pass(self):
        self.assertFalse(evals.reportable([]))
        self.assertFalse(evals.reportable([{"name": "execution_ok", "passed": True}]))

    def test_integrity_failure_blocks_retry_even_with_lookahead_failure(self):
        result = runner.run_local_demo("initial")
        result["metrics"]["total_return"] += 0.2
        checks = evals.evaluate_result(result)
        self.assertFalse(evals.reportable(checks))
        self.assertFalse(evals.retry_eligible(checks))

    def test_malformed_metrics_fail_without_exception(self):
        valid = runner.run_local_demo("repaired")
        for value in [None, True, float("nan"), float("inf"), 10 ** 1000, "1"]:
            result = copy.deepcopy(valid)
            result["metrics"]["total_return"] = value
            self.assertFalse(evals.reportable(evals.evaluate_result(result)))

    def test_price_mutation_is_rejected(self):
        def mutator(prices):
            prices[0] = 1
            return [0] * len(prices)
        with self.assertRaisesRegex(ValueError, "mutated"):
            harness.evaluate([10, 11, 12], mutator)

    def test_stateful_signals_fail_consistency(self):
        calls = 0
        def unstable(prices):
            nonlocal calls
            calls += 1
            return [calls % 2] * len(prices)
        self.assertFalse(harness.evaluate([10, 11, 12, 13], unstable)["lookahead"]["passed"])

    def test_strategy_parser_accepts_annotations_and_rejects_bad_contracts(self):
        annotated = "def signals(prices: list[float]) -> list[int]:\n    return [0] * len(prices)"
        self.assertEqual(agent._extract_strategy(f"```python\n{annotated}\n```"), annotated)
        for source in ["def signals(data):\n return []", "def signals(prices, extra):\n return []", "def signals(prices):\n return ["]:
            with self.assertRaises((ValueError, SyntaxError)):
                agent._extract_strategy(source)

    def test_anthropic_real_sdk_request_and_response(self):
        import anthropic
        import httpx2
        requests = []
        def transport(request):
            requests.append(json.loads(request.content))
            return httpx2.Response(200, json={
                "id": "message-audit", "type": "message", "role": "assistant", "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": MOMENTUM_SOURCE}], "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 10},
            })
        client = anthropic.Anthropic(api_key="test-only", http_client=httpx2.Client(transport=httpx2.MockTransport(transport)))
        with patch("anthropic.Anthropic", return_value=client) as constructor:
            source, _ = agent._repair_with_anthropic("momentum", "future leakage", "source note", LEAKY_SOURCE)
        self.assertEqual(source, MOMENTUM_SOURCE.strip())
        self.assertIn(LEAKY_SOURCE, requests[0]["messages"][0]["content"])
        constructor.assert_called_once_with(timeout=30, max_retries=0)

    def test_bedrock_converse_contract(self):
        import boto3
        from botocore.stub import Stubber
        os.environ.update(AWS_BEDROCK_MODEL_ID="audit-model", AWS_REGION="us-east-1")
        client = boto3.client("bedrock-runtime", region_name="us-east-1", aws_access_key_id="test-only", aws_secret_access_key="test-only")
        expected = {
            "modelId": "audit-model", "system": [{"text": agent.SYSTEM_PROMPT}],
            "messages": [{"role": "user", "content": [{"text": agent._repair_prompt("momentum", "leak", "note", LEAKY_SOURCE)}]}],
            "inferenceConfig": {"maxTokens": 700, "temperature": 0.2},
        }
        response = {
            "output": {"message": {"role": "assistant", "content": [{"text": MOMENTUM_SOURCE}]}},
            "stopReason": "end_turn", "usage": {"inputTokens": 10, "outputTokens": 10, "totalTokens": 20},
            "metrics": {"latencyMs": 1},
        }
        with Stubber(client) as stubber:
            stubber.add_response("converse", response, expected)
            with patch("boto3.client", return_value=client):
                source, _ = agent._repair_with_bedrock("momentum", "leak", "note", LEAKY_SOURCE)
            stubber.assert_no_pending_responses()
        self.assertEqual(source, MOMENTUM_SOURCE.strip())

    def test_repair_provider_fallback(self):
        os.environ.update(AWS_BEDROCK_MODEL_ID="test", ANTHROPIC_API_KEY="test")
        with patch.object(agent, "_repair_with_bedrock", side_effect=RuntimeError), patch.object(agent, "_repair_with_anthropic", return_value=(MOMENTUM_SOURCE, "ok")):
            self.assertEqual(agent.repair_strategy("h", "f")[2], "anthropic_api")
        with patch.object(agent, "_repair_with_bedrock", side_effect=RuntimeError), patch.object(agent, "_repair_with_anthropic", side_effect=RuntimeError):
            self.assertEqual(agent.repair_strategy("h", "f")[2], "trusted_fallback")

    def test_daytona_parsing_and_cleanup(self):
        os.environ["DAYTONA_API_KEY"] = "test-only"
        client = Mock()
        sandbox = client.create.return_value
        good = runner.run_local_demo("repaired")
        with patch("daytona.Daytona", return_value=client):
            for code, output, success in [(0, "RESULT_JSON:" + json.dumps(good), True),
                                          (1, "RESULT_JSON:" + json.dumps(good), False),
                                          (0, 'RESULT_JSON:{}\nRESULT_JSON:{}', False),
                                          (0, 'RESULT_JSON:[]', False)]:
                sandbox.process.exec.return_value = SimpleNamespace(exit_code=code, result=output)
                self.assertEqual(runner.run_daytona(MOMENTUM_SOURCE).get("ok"), success)
                client.delete.assert_called_with(sandbox, wait=True, timeout=30)
            sandbox.fs.upload_file.side_effect = RuntimeError("upload failed")
            self.assertFalse(runner.run_daytona(MOMENTUM_SOURCE)["ok"])
            self.assertEqual(client.delete.call_count, 5)

    def test_anchor_real_sdk_wire_contract_and_cleanup(self):
        from anchorbrowser import Anchorbrowser
        os.environ["ANCHORBROWSER_API_KEY"] = "test-only"
        requests = []
        def transport(request):
            requests.append(request)
            if request.method == "DELETE":
                return httpx.Response(200, json={"success": True})
            if "perform-web-task" in request.url.path:
                return httpx.Response(200, json={"data": {"result": "Public page facts"}})
            return httpx.Response(200, json={"data": {"id": "audit-session"}})
        client = Anchorbrowser(api_key="test-only", http_client=httpx.Client(transport=httpx.MockTransport(transport)))
        with patch("anchorbrowser.Anchorbrowser", return_value=client):
            self.assertEqual(sponsors.research_with_anchor("https://example.com", "momentum"), "Public page facts")
        self.assertEqual([request.method for request in requests], ["POST", "POST", "DELETE"])
        body = json.loads(requests[1].content)
        self.assertEqual(body["max_steps"], 5)
        self.assertEqual(body["url"], "https://example.com")
        self.assertFalse(body["human_intervention"])
        self.assertEqual(requests[1].extensions["timeout"]["read"], 45)
        self.assertTrue(client.is_closed())

    def test_anchor_preserves_api_failure_and_deletes_session(self):
        from anchorbrowser import Anchorbrowser, APIStatusError
        os.environ["ANCHORBROWSER_API_KEY"] = "test-only"
        requests = []
        def transport(request):
            requests.append(request)
            if request.method == "DELETE":
                return httpx.Response(500, json={"error": "cleanup failed"})
            if "perform-web-task" in request.url.path:
                return httpx.Response(402, json={"error": "Insufficient credits balance"})
            return httpx.Response(200, json={"data": {"id": "audit-session"}})
        client = Anchorbrowser(api_key="test-only", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(transport)))
        with patch("anchorbrowser.Anchorbrowser", return_value=client), self.assertRaises(APIStatusError) as caught:
            sponsors.research_with_anchor("https://example.com", "momentum")
        self.assertEqual(caught.exception.status_code, 402)
        self.assertEqual(requests[-1].method, "DELETE")
        self.assertTrue(client.is_closed())

    def test_anchor_rejects_local_and_credential_urls(self):
        for url in ["http://localhost:8503", "http://127.0.0.1", "http://[::1]", "http://service.internal", "https://user:password@example.com", "https://example.com:invalid"]:
            with self.assertRaises(ValueError):
                sponsors.research_with_anchor(url, "test")

    def test_surreal_context_schema_and_write_acknowledgement(self):
        with patch.object(sponsors, "_surreal_request", return_value={"context": "prior note"}) as call:
            self.assertEqual(sponsors.recall_prior_runs("momentum"), "prior note")
            self.assertEqual(call.call_args.args[0], "context")
        with patch.object(sponsors, "_surreal_request", return_value={"chunkId": "stored-chunk"}) as call:
            sponsors.remember_run({"run_id": "audit-run"})
            self.assertEqual(call.call_args.kwargs["idempotency_key"], "audit-run")
            self.assertEqual(call.call_args.args[1]["infer"], "none")
        with patch.object(sponsors, "_surreal_request", return_value={}):
            with self.assertRaises(RuntimeError):
                sponsors.remember_run({})
            with self.assertRaises(RuntimeError):
                sponsors.recall_prior_runs("test")

    def test_surreal_auth_header_and_host_validation(self):
        os.environ.update(SPECTRON_URL="https://memory.example.com", SPECTRON_CONTEXT_ID="context/a", SPECTRON_API_KEY="test-only")
        opener = Mock()
        opener.open.return_value = io.BytesIO(b'{"context":"note"}')
        with patch.object(sponsors, "build_opener", return_value=opener):
            self.assertEqual(sponsors.recall_prior_runs("momentum"), "note")
        request = opener.open.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer test-only")
        self.assertEqual(request.full_url, "https://memory.example.com/api/v1/context%2Fa/context")
        os.environ["SPECTRON_URL"] = "http://memory.example.com"
        with self.assertRaises(ValueError):
            sponsors.recall_prior_runs("test")
        with self.assertRaises(RuntimeError):
            sponsors._NoCredentialRedirects().redirect_request(None, None, 302, "", {}, "https://other.example")

    def test_tracer_identity_isolated_across_runs(self):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"
        providers, exporters = [], []
        for _ in range(2):
            provider, exporter = TracerProvider(), InMemorySpanExporter()
            provider.add_span_processor(SimpleSpanProcessor(exporter))
            providers.append(provider)
            exporters.append(exporter)
        with patch.object(tracing, "load_phoenix_environment"), patch("phoenix.otel.register", side_effect=providers) as register:
            for index in range(2):
                tracer, provider = tracing.make_tracer()
                with tracer.start_as_current_span(f"run-{index}"):
                    pass
                provider.force_flush()
            self.assertFalse(register.call_args.kwargs["set_global_tracer_provider"])
            self.assertEqual(register.call_args.kwargs["endpoint"], "http://localhost:6006/v1/traces")
            self.assertEqual(register.call_args.kwargs["protocol"], "http/protobuf")
        for index, exporter in enumerate(exporters):
            self.assertEqual([span.name for span in exporter.get_finished_spans()], [f"run-{index}"])
            providers[index].shutdown()

    def test_streamlit_form_to_gate_and_memory(self):
        from streamlit.testing.v1 import AppTest
        os.environ.update(SPECTRON_URL="https://memory.example.com", SPECTRON_CONTEXT_ID="test", SPECTRON_API_KEY="test-only")
        root = Path(__file__).resolve().parents[1]
        with patch("dotenv.load_dotenv"), patch.object(tracing, "load_phoenix_environment"), patch.object(sponsors, "recall_prior_runs", return_value="Prior causal repair note"), patch.object(sponsors, "remember_run") as remember:
            app = AppTest.from_file(str(root / "app.py")).run()
            self.assertFalse(app.exception)
            app.button[0].click().run(timeout=10)
            self.assertFalse(app.exception)
            self.assertTrue(app.success)
            record = remember.call_args.args[0]
            self.assertTrue(record["final_passed"])
            self.assertEqual(record["attempt_count"], 2)
            self.assertEqual(record["repair_mode"], "trusted_local_demo")
            first_id = record["run_id"]
            app.button[0].click().run(timeout=10)
            self.assertFalse(app.exception)
            self.assertNotEqual(remember.call_args.args[0]["run_id"], first_id)


if __name__ == "__main__":
    unittest.main()
