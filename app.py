from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

import streamlit as st
from opentelemetry.trace import Status, StatusCode

from strategy_lab.agent import repair_strategy
from strategy_lab.evals import evaluate_result, reportable, retry_eligible
from strategy_lab.runner import run_daytona, run_local_demo
from strategy_lab.sponsor_integrations import (
    anchor_configured,
    recall_prior_runs,
    remember_run,
    research_with_anchor,
    surreal_configured,
)
from strategy_lab.strategies import LEAKY_SOURCE, MOMENTUM_SOURCE
from strategy_lab.tracing import load_phoenix_environment, make_tracer


def show_attempt_checks(label: str, checks: list[dict]) -> None:
    passed = sum(check["passed"] for check in checks)
    st.write(f"**{label}: {passed}/{len(checks)} checks passed**")
    for check in checks:
        marker = "✅" if check["passed"] else "❌"
        st.write(f"{marker} **{check['name'].replace('_', ' ').capitalize()}** — {check['detail']}")


st.set_page_config(page_title="Strategy Lab", page_icon="🛡️", layout="wide")
load_dotenv(Path(__file__).resolve().parent / ".env")
load_phoenix_environment()
st.markdown("""
<style>
:root {--lab-ink:#f5f5f2;--lab-muted:#a1a1a1;--lab-line:#393939;--lab-accent:#eeeeea;}
.block-container {max-width:1200px;padding:4rem 2rem 2rem;}
[data-testid="stAppViewContainer"], [data-testid="stHeader"] {background:#101010;color:var(--lab-ink);}
[data-testid="stSidebar"] {background:#161616;border-right:1px solid var(--lab-line);}
[data-testid="stVerticalBlock"] {gap:.8rem;}
h1,h2,h3 {color:var(--lab-ink);letter-spacing:-.025em;line-height:1.05;}
h1 {font-size:clamp(2.2rem,4vw,3.4rem)!important;font-weight:600!important;padding:.2rem 0 .6rem!important;}
h2 {font-size:1.65rem!important;} h3 {font-size:1.4rem!important;}
p, label {color:var(--lab-ink);}
[data-testid="stCaptionContainer"] p {color:var(--lab-muted);font-size:.78rem;line-height:1.5;}
[data-testid="stForm"] {background:#141414;border:1px solid var(--lab-line);border-radius:0;padding:20px;}
[data-testid="stMetric"] {background:transparent;border:0;border-left:1px solid var(--lab-line);border-radius:0;padding:8px 16px;}
[data-testid="stMetricValue"],code,pre,input,.lab-index,.lab-process,[data-testid="stMetricLabel"] {font-family:"SFMono-Regular",Consolas,"Liberation Mono",monospace!important;}
[data-testid="stMetricValue"] {font-size:2rem;letter-spacing:-.06em;}
[data-baseweb="input"], [data-baseweb="base-input"], [data-testid="stExpander"], [data-testid="stExpander"] details {background:#101010!important;border-radius:0!important;border-color:var(--lab-line)!important;}
input {color:var(--lab-ink)!important;font-size:.88rem!important;}
button, [data-testid="stLinkButton"] a {border-radius:2px!important;box-shadow:none!important;}
.stButton>button,.stFormSubmitButton>button,[data-testid="stLinkButton"] a {border:1px solid #555;background:#171717;color:var(--lab-ink);}
.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"] {background:var(--lab-accent);color:#101010;border:1px solid var(--lab-accent);font-weight:750;}
.stFormSubmitButton>button[kind="primary"] p {color:#101010!important;}
[data-testid="stFormSubmitButton"] button {background:var(--lab-accent)!important;border-color:var(--lab-accent)!important;color:#101010!important;}
[data-testid="stFormSubmitButton"] button p {color:#101010!important;font-weight:750;}
[data-testid="stTextInput"] div {border-radius:0!important;background:#101010!important;}
button:hover {border-color:var(--lab-accent)!important;}
button:focus-visible,input:focus-visible {outline:2px solid var(--lab-accent)!important;outline-offset:3px;}
[data-testid="stAlert"] {background:#191919!important;border:1px solid #555;border-left:3px solid var(--lab-accent);border-radius:0;}
.lab-index {display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid var(--lab-line);padding-bottom:10px;color:var(--lab-accent);font-size:.69rem;letter-spacing:.07em;}
.lab-subtitle {font-size:1rem;color:#b4b4b4;max-width:650px;margin:0 0 18px;line-height:1.5;}
.lab-process {display:flex;flex-wrap:wrap;gap:0;border-top:1px solid var(--lab-line);border-bottom:1px solid var(--lab-line);margin-bottom:18px;font-size:.73rem;color:#b4b4b4;}
.lab-process span {padding:11px 16px 11px 0;margin-right:20px;}
.lab-process b {color:var(--lab-accent);margin-right:8px;font-weight:500;}
hr {border-color:var(--lab-line);}
@media(max-width:700px) {.block-container {padding:4rem 1rem 1.5rem;} .lab-index {font-size:.59rem;} .lab-process {display:block;} .lab-process span {display:block;padding:7px 0;margin:0;} [data-testid="stForm"] {padding:14px;}}
</style>
<div class="lab-index"><span>RESEARCH WORKSPACE</span><span>STRATEGY LAB</span></div>
""", unsafe_allow_html=True)
st.title("Strategy Lab")
st.markdown('<div class="lab-subtitle">Evaluate a strategy, inspect its failures, and review the corrected result.</div><div class="lab-process"><span>Candidate</span><span>Evaluation &amp; correction</span><span>Report</span></div>', unsafe_allow_html=True)

daytona_ready = bool(os.getenv("DAYTONA_API_KEY"))
phoenix_ready = bool(os.getenv("PHOENIX_COLLECTOR_ENDPOINT"))
arize_ax_ready = bool(os.getenv("ARIZE_SPACE_ID") and os.getenv("ARIZE_API_KEY"))
anthropic_ready = bool(os.getenv("ANTHROPIC_API_KEY"))
bedrock_ready = bool(os.getenv("AWS_BEDROCK_MODEL_ID"))
anchor_ready = anchor_configured()
surreal_ready = surreal_configured()
mode = "Daytona sandbox" if daytona_ready else "Local demo (trusted built-in strategies only)"
observability = "Phoenix local" if phoenix_ready else "Arize AX Cloud" if arize_ax_ready else "not configured"
repair_status = "Amazon Bedrock selected" if bedrock_ready else "Anthropic API selected" if anthropic_ready else "built-in demo repair"
anchor_status = "Anchor Browser key set" if anchor_ready else "Anchor Browser not connected"
memory_status = "SurrealDB credentials set" if surreal_ready else "SurrealDB memory not connected"
with st.sidebar:
    st.header("Environment")
    st.write(f"**Execution**  \n{mode}")
    st.write(f"**Tracing**  \n{observability}")
    st.write(f"**Repair provider**  \n{repair_status}")
    st.divider()
    st.caption("Optional integrations · configuration is not proof of a successful request")
    st.write(anchor_status)
    st.write(memory_status)
    st.caption("Actual connection results appear during each run.")
    if phoenix_ready:
        st.link_button("Open Phoenix traces", "http://localhost:6006")

st.subheader("Strategy evaluation")
st.caption("Seeded example / Synthetic prices / One correction")
if not daytona_ready:
    st.info("Rehearsal mode: only trusted, built-in strategies run locally.")
if not (bedrock_ready or anthropic_ready):
    st.caption("Repair uses a fixed causal baseline. This run does not use an AI model to write code.")

with st.form("run_form"):
    hypothesis = st.text_input("Demo hypothesis", "Follow the previous price move: long after a flat or rising close, short after a falling close.")
    st.caption("Signals take effect on the next bar, with transaction costs. The first candidate is a fixed injected bug; editing this hypothesis does not generate a new first candidate.")
    with st.expander("Add a public research source · optional"):
        research_url = st.text_input("Public research URL for Anchor Browser", "")
    submitted = st.form_submit_button("Run safety demo", type="primary")

if submitted:
    run_id = str(uuid4())
    try:
        tracer, provider = make_tracer()
        tracing_error = None
    except Exception as error:
        tracer, provider = None, None
        tracing_error = str(error)
    if tracing_error:
        st.warning(f"Tracing could not start for this run: {tracing_error}")
    trace_context = None
    span = None
    trace_flushed = False
    try:
        trace_context = tracer.start_as_current_span("backtest_agent_run") if tracer else None
        span = trace_context.__enter__() if trace_context else None
        if span:
            span.set_attribute("run.id", run_id)
            span.set_attribute("strategy.hypothesis", hypothesis)
            span.set_attribute("execution.mode", mode)
            span.set_attribute("execution.sandbox_isolated", daytona_ready)
            span.set_attribute("integration.anchor_enabled", anchor_ready and bool(research_url.strip()))
            span.set_attribute("integration.surreal_memory_enabled", surreal_ready)
            if daytona_ready:
                span.set_attribute("execution.network_block_all", True)
                span.set_attribute("execution.cpu_limit", 1)
                span.set_attribute("execution.memory_gib_limit", 2)
                span.set_attribute("execution.disk_gib_limit", 8)
                span.set_attribute("execution.timeout_seconds", 45)
        with st.status("Running the seeded failure through the evaluator…", expanded=True) as status:
            source_evidence = ""
            prior_memory = ""
            if research_url.strip():
                if anchor_ready:
                    try:
                        source_evidence = research_with_anchor(research_url, hypothesis)
                        st.write("**Anchor Browser source notes**")
                        st.write(source_evidence)
                        st.caption(f"Source page: {research_url.strip()}")
                        if span:
                            span.set_attribute("integration.anchor_succeeded", True)
                    except Exception as error:
                        if getattr(error, "status_code", None) == 402:
                            st.warning("Anchor Browser reports an insufficient credit balance. The rest of the demo will continue; add or claim Anchor credits to enable this source-research step.")
                        else:
                            st.warning(f"Anchor research was skipped ({type(error).__name__}). The rest of the demo will continue.")
                        if span:
                            span.set_attribute("integration.anchor_succeeded", False)
                else:
                    st.warning("Add ANCHORBROWSER_API_KEY to use the optional Anchor source step.")
            if surreal_ready:
                try:
                    prior_memory = recall_prior_runs(hypothesis)
                    st.write("**SurrealDB Agent Memory**")
                    st.write("Recalled prior run notes." if prior_memory else "No related run notes yet; this run will seed the memory.")
                    if prior_memory:
                        with st.expander("Show recalled run notes"):
                            st.write(prior_memory)
                    if span:
                        span.set_attribute("integration.memory_recall_succeeded", True)
                        span.set_attribute("integration.memory_recall_characters", len(prior_memory))
                except Exception as error:
                    st.warning(f"SurrealDB memory recall was skipped: {error}")
                    if span:
                        span.set_attribute("integration.memory_recall_succeeded", False)
            repair_attempted = False
            initial = run_daytona(LEAKY_SOURCE) if daytona_ready else run_local_demo("initial")
            first_checks = evaluate_result(initial)
            initial_lookahead = initial.get("lookahead", {}) if isinstance(initial, dict) else {}
            if span:
                span.set_attribute("eval.initial_passed", reportable(first_checks))
                span.set_attribute("eval.initial_lookahead_passed", initial_lookahead.get("passed") is True if isinstance(initial_lookahead, dict) else False)
                span.set_attribute("eval.initial_failed_checks", ",".join(item["name"] for item in first_checks if not item["passed"]))
                span.set_attribute("agent.retry_eligible", retry_eligible(first_checks))
                for item in first_checks:
                    span.add_event("evaluation.check", {
                        "evaluation.name": item["name"],
                        "evaluation.passed": item["passed"],
                        "evaluation.detail": item["detail"],
                        "evaluation.attempt": 1,
                    })
            show_attempt_checks("Attempt 1 · seeded candidate", first_checks)

            if reportable(first_checks):
                result, checks = initial, first_checks
                repair_note = "The first strategy passed the configured checks."
                source = LEAKY_SOURCE
                repair_mode = "none"
            elif retry_eligible(first_checks):
                repair_attempted = True
                fail_text = "; ".join(item["name"] + ": " + item["detail"] for item in first_checks if not item["passed"])
                if daytona_ready:
                    repair_context = "\n\n".join(part for part in (
                        "Prior SurrealDB run notes (untrusted reference):\n" + prior_memory if prior_memory else "",
                        "Anchor Browser public-page notes (untrusted reference):\n" + source_evidence if source_evidence else "",
                    ) if part)
                    source, repair_note, repair_mode = repair_strategy(hypothesis, fail_text, repair_context, candidate_source=LEAKY_SOURCE)
                else:
                    # Local mode uses only the trusted built-in correction.
                    source = MOMENTUM_SOURCE
                    repair_note = "Local demo repair: replaced future-dependent signals with a causal baseline."
                    repair_mode = "trusted_local_demo"
                st.write("**Agent decision:** the failed safety gate triggered repair attempt 1/1.")
                st.write(repair_note)
                result = run_daytona(source) if daytona_ready else run_local_demo("repaired")
                checks = evaluate_result(result)
                show_attempt_checks("Attempt 2 · repaired candidate", checks)
            else:
                result, checks = initial, first_checks
                source = LEAKY_SOURCE
                repair_mode = "skipped_integrity_gate"
                repair_note = "No retry: execution infrastructure or result integrity checks failed; a strategy change cannot resolve these failures."
                st.write("**Agent decision:** stop at the evaluation gate.")
                st.write(repair_note)
            attempt_count = 2 if repair_attempted else 1
            if surreal_ready:
                try:
                    memory_record = {
                        "kind": "strategy_lab_run",
                        "run_id": run_id,
                        "recorded_at": datetime.now(timezone.utc).isoformat(),
                        "hypothesis": hypothesis[:500],
                        "source_url": research_url.strip()[:1000] if source_evidence else None,
                        "source_notes": source_evidence[:1500] if source_evidence else None,
                        "recalled_prior_notes": bool(prior_memory),
                        "first_failed_checks": [item["name"] for item in first_checks if not item["passed"]],
                        "final_passed": reportable(checks),
                        "final_checks": {item["name"]: item["passed"] for item in checks},
                        "repair_mode": repair_mode,
                        "attempt_count": attempt_count,
                        "metrics": result.get("metrics") if isinstance(result, dict) and reportable(checks) else None,
                    }
                    remember_run(memory_record)
                    st.caption("This run was saved to SurrealDB Agent Memory.")
                    if span:
                        span.set_attribute("integration.memory_write_succeeded", True)
                except Exception as error:
                    st.warning(f"This run could not be saved to SurrealDB: {error}")
                    if span:
                        span.set_attribute("integration.memory_write_succeeded", False)
            if span:
                span.set_attribute("eval.final_passed", reportable(checks))
                span.set_attribute("eval.final_checks", ",".join(item["name"] + ":" + str(item["passed"]) for item in checks))
                span.set_attribute("agent.repair_mode", repair_mode)
                span.set_attribute("agent.repair_note", repair_note)
                span.set_attribute("execution.attempt_count", attempt_count)
                for item in checks:
                    span.add_event("evaluation.check", {
                        "evaluation.name": item["name"],
                        "evaluation.passed": item["passed"],
                        "evaluation.detail": item["detail"],
                        "evaluation.attempt": attempt_count,
                    })
            status.update(label="All checks passed · report unlocked" if reportable(checks) else "Checks failed · report withheld", state="complete" if reportable(checks) else "error", expanded=False)
        st.session_state["last_verdict"] = {"run_id": run_id, "passed": reportable(checks), "attempts": attempt_count, "repair": repair_mode}
        st.subheader("Evaluation verdict")
        with st.expander("Inspect all evaluation checks"):
            show_attempt_checks("Final candidate", checks)
        if reportable(checks):
            metrics = result["metrics"]
            st.success("Reportable: execution completed and all configured checks passed. This does not establish profitability.")
            columns = st.columns(4)
            columns[0].metric("Simulated return", f"{metrics['total_return']:.2%}")
            columns[1].metric("Maximum drawdown", f"{metrics['max_drawdown']:.2%}")
            columns[2].metric("Position changes", str(metrics["trades"]))
            columns[3].metric("Sharpe (per bar)", f"{metrics['sharpe']:.2f}")
            st.caption("Synthetic demo equity curve")
            st.line_chart(metrics["equity"])
            st.caption("The sample series is synthetic. These figures demonstrate the evaluator and do not establish profitability or predict live performance.")
            with st.expander("Show repaired strategy code"):
                st.code(source, language="python")
        else:
            st.error("The agent could not produce a report that passed the evaluation gate.")
            if not isinstance(result, dict):
                st.code("Harness result was not a JSON object")
            elif not result.get("ok"):
                st.code(result.get("error", "Unknown execution error"))
    except Exception as error:
        if span:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR))
        st.error(f"Run failed: {error}")
    finally:
        if trace_context and span:
            try:
                trace_context.__exit__(None, None, None)
            except Exception:
                st.warning("The run ended, but its trace could not be closed cleanly.")
        if provider:
            try:
                trace_flushed = provider.force_flush(timeout_millis=2000) is not False
            except Exception as error:
                st.warning(f"The run completed, but trace export failed: {error}")
            finally:
                try:
                    provider.shutdown()
                except Exception:
                    st.warning("The tracing connection could not be closed cleanly.")
    if trace_flushed and phoenix_ready:
        st.caption(f"Phoenix project: {os.getenv('PHOENIX_PROJECT_NAME', 'strategy-lab')} · run ID: {run_id}. Open Phoenix to confirm receipt.")
    elif trace_flushed and arize_ax_ready:
        st.caption(f"Arize AX project: trustworthy-backtest-agent · run ID: {run_id}. Open Arize to confirm receipt.")

if not submitted and "last_verdict" in st.session_state:
    previous = st.session_state["last_verdict"]
    st.caption(f"Previous run: {'passed' if previous['passed'] else 'withheld'} · {previous['attempts']} attempts · {previous['repair']} · {previous['run_id']}")

with st.expander("How the demo works · integrations and limits"):
    st.markdown(
        "- **Daytona:** runs the candidate in an ephemeral sandbox with outbound network blocking requested, 1 CPU, 2 GiB memory, 8 GiB disk, and a 45-second execution timeout.\n"
        "- **AWS:** Amazon Bedrock can generate one repair through Converse. AWS credentials stay in the app process and are not passed into Daytona. The trace records which repair path ran.\n"
        "- **Agent behavior:** uses evaluation feedback to request one bounded repair.\n"
        "- **Anchor Browser:** optionally reads a public research page and provides untrusted context for the repair; it does not submit forms.\n"
        "- **SurrealDB Agent Memory:** optionally recalls prior runs and stores each final evaluation for future repairs.\n"
        "- **Tracing:** Phoenix OSS or Arize AX Cloud records the run and pass/fail evaluation attributes.\n"
        "- **Safety boundary:** only the Daytona path executes model-generated source; local demo mode calls trusted built-in functions. The requested Daytona resource and network settings are recorded in the trace; the containment probe checks observed behavior."
    )

st.divider()
st.subheader("Sandbox containment")
st.caption("Controlled probe / Daytona sandbox")
st.write("A fixed probe checks whether named application secrets are present and attempts one outbound connection. It never reads secret values or sends application data.")
if st.button("Run containment demo", disabled=not daytona_ready, key="containment_demo"):
    from strategy_lab.containment import run_containment_demo
    with st.spinner("Creating an isolated sandbox, running the probe, and deleting it…"):
        st.session_state["containment_result"] = run_containment_demo()
if "containment_result" in st.session_state:
    evidence = st.session_state["containment_result"]
    if evidence.get("ok"):
        left, right = st.columns(2)
        left.metric("Application secrets", "Absent" if evidence["secrets_absent"] else "Unexpectedly present")
        right.metric("Outbound connection", "Succeeded" if evidence["network_connected"] else "Unsuccessful")
        if evidence["secrets_absent"] and not evidence["network_connected"]:
            st.success("The probe found no named application secrets and could not connect outward. Sandbox deleted.")
        else:
            st.error("The probe found an unexpected result. Inspect the evidence before presenting.")
    else:
        st.error(evidence.get("error", "No usable probe result"))
    with st.expander("Inspect containment evidence"):
        st.json(evidence)
st.caption("A failed connection alone does not prove all networking is blocked. This probe runs with Daytona’s network-blocking policy; it does not establish tamper-proof backtest results.")
