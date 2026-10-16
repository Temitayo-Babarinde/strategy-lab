# Strategy Lab

A one-day hackathon demo: when Daytona is configured, a strategy candidate runs in an isolated sandbox; a deterministic prefix check looks for future-data leakage; and the agent makes one bounded repair after a failure. Amazon Bedrock can generate that repair through its Converse API. Without the Daytona key, the app uses trusted built-in functions for local rehearsal. Runs can be traced to local Phoenix or, when configured, Arize AX Cloud.

The included price series is synthetic and exists only to make the demo repeatable. A passing check means this lookahead check and the result-shape checks passed; it does not mean a strategy is profitable or safe for live trading. The Sharpe display is per bar and unannualized because the sample does not define a reliable bar frequency.

## Start the local demo

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Use your own price data

Upload a UTF-8 CSV with a `close` column and at least three finite, positive prices using the **Price data CSV** control. A `timestamp` column is optional. Other columns appear in the preview but are ignored by the current backtest. If Daytona is configured, the uploaded CSV is sent to Daytona for sandbox execution; in local rehearsal mode it stays on your computer. The app validates the data before enabling a run and shows the first ten rows as a preview.

With no credentials, the app runs only trusted, built-in demonstration functions locally. It labels this mode in the UI and does not execute generated code on your computer.

## Connect Phoenix and the event services

For local Phoenix tracing, start `uvx arize-phoenix serve` in one terminal and run `npx -y @arizeai/phoenix-cli setup` from this project in another. The setup writes `.env.phoenix`; the app reads that file and sends traces to the local Phoenix project without an API key. To keep traces in this project and limit the server to your computer, set `PHOENIX_WORKING_DIR="$PWD/.phoenix-data"` and `PHOENIX_HOST=127.0.0.1` before starting it.

For the event's Arize AX Cloud workspace instead, set `ARIZE_SPACE_ID` and `ARIZE_API_KEY`. Local Phoenix and AX Cloud are separate destinations; the app uses the Phoenix config when it is present and otherwise falls back to AX Cloud credentials.

Optional sponsor integrations:

- **Anchor Browser:** set `ANCHORBROWSER_API_KEY`, enter a public research URL, and the agent reads the page as untrusted context before repair. It does not sign in or submit forms. Install the Python SDK with the requirements above. A key alone is not enough if the account has no service credits; a 402 response skips this optional step while the rest of the demo continues.
- **SurrealDB Agent Memory:** set `SPECTRON_URL`, `SPECTRON_CONTEXT_ID`, and `SPECTRON_API_KEY` from your context's API Keys setup. The app recalls up to three related notes before repair and stores the final checks and metrics after each run. Writes use `infer: none`; memory failures do not stop the evaluation.

Copy `.env.example` to `.env` and add `DAYTONA_API_KEY` to run candidate code in Daytona. For AWS, configure credentials accessible to the app process through the AWS SDK credential chain (such as an AWS profile or temporary environment credentials), then set `AWS_REGION` and an enabled `AWS_BEDROCK_MODEL_ID` that supports Converse. The caller needs `bedrock:InvokeModel` permission. Bedrock repairs take priority; Anthropic is the next fallback, followed by the trusted built-in repair. AWS credentials stay in the app process and are not passed into Daytona. Generated candidates run only in Daytona. `ANTHROPIC_API_KEY` is optional. The app loads `.env` locally, and `.gitignore` keeps it out of source control. The page labels local rehearsal mode clearly and warns when Daytona or tracing credentials are missing.

Restart Streamlit after changing credentials in `.env`; existing process or shell settings take priority. API-sensitive dependencies are pinned to the versions checked in the October 16 code review.

When Daytona is configured, the app creates an ephemeral Python 3.12 sandbox with 1 CPU, 2 GiB memory, and 8 GiB disk, requests outbound network blocking, uploads the small CSV and standard-library harness, and runs the candidate with a 45-second process timeout. It parses one JSON result and deletes the sandbox afterward. The Arize/Phoenix trace records the execution mode, sandbox limits, evaluation results, and repair path. Daytona resource limits are also subject to the event workspace's organization limits.

## Demo sequence

1. Enter a short hypothesis and click **Run safety demo**.
2. Show the seeded candidate fail because its signal changes when later price rows are hidden.
3. Show the agent receive that failure and select a built-in causal baseline (or request a model-generated repair when a provider is configured).
4. Show the passing checks and the Arize trace.

## Current limits

- The live containment probe observed a successful outbound TCP connection despite the requested network block. Network containment is not verified; keep secrets out of candidate sandboxes.


- One synthetic dataset and one seeded failure keep the result repeatable.
- The prefix check catches a useful class of lookahead bugs, not every possible flaw.
- The metrics are illustrative and are not used to claim profitability.
- Local mode is for rehearsing the story. Use Daytona for sandbox execution; choose Phoenix OSS or Arize AX Cloud for tracing.
- The candidate and harness share a sandbox. These checks address ordinary strategy bugs and malformed results; they are not tamper-proof attestation against deliberately malicious candidate code.

## Repeat the offline review checks

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
```

The tests cover the evaluation gate, source consistency, model repair request formats, Daytona error handling, Anchor session cleanup, SurrealDB acknowledgement, tracing identity, and the Streamlit form flow. They use fake credentials and local SDK transports rather than live API calls.
