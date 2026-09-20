# InferPulse Bench

[简体中文](README.md) · [English](README.en.md)

A standalone Python benchmark for locally deployed LLM services. Configure one model, check its thinking-mode behavior, test input sizes and concurrency levels, and generate a report backed by request-level evidence.

**Source version 1.9.0 · Python 3.9+ · Standard library only · MIT**

[Download](https://gitee.com/xum1983/inferpulse-bench/releases/tag/v1.9.0) · [Markdown sample report](https://gitee.com/xum1983/inferpulse-bench/blob/master/report.example.md) · [HTML report](https://gitee.com/xum1983/inferpulse-bench/blob/master/report.example.html) · [PDF report](https://gitee.com/xum1983/inferpulse-bench/blob/master/report.example.pdf) · [Contributing](https://gitee.com/xum1983/inferpulse-bench/blob/master/CONTRIBUTING.md) · [☕ Support the author](https://gitee.com/xum1983/inferpulse-bench/blob/master/SPONSOR.md)

CLI messages, generated reports and optional model self-review are currently in **Chinese**. This README provides English instructions; it does not enable an English output mode.

## Project scope

| Product | Repository | Status |
|---|---|---|
| **InferPulse Bench** | This repository, `inferpulse-bench` | Standalone script; available now |
| **InferPulse** | [`inferpulse`](https://gitee.com/xum1983/inferpulse) | Full graphical product; placeholder repository reserved, code and installers not yet publicly released |

The products have separate versions and releases. Their configuration and evidence formats are not currently interchangeable. This repository's MIT license applies to Bench; the full product will state its license separately.

The source and packaged release are both version 1.9.0, generating Markdown and HTML by default with configurable formats. Default benchmark settings and metric definitions are unchanged.

## Quick start

You need Python 3.9 or newer and network access to the target service. No `pip install`, activation key or runtime expiry is required.

1. Download **`inferpulse-bench-1.9.0.zip`** from the [release page](https://gitee.com/xum1983/inferpulse-bench/releases/tag/v1.9.0) and extract it. It includes an empty-key `llm_benchmark.jsonc`. `SHA256SUMS.txt` is available alongside the ZIP.
2. If using a source checkout instead, copy either `llm_benchmark.deepseek.example.jsonc` or `llm_benchmark.qwen.example.jsonc` to `llm_benchmark.jsonc` next to the script.
3. Edit `model.name`, `model.api_url` and `model.api_key`. Use the exact model name accepted by the service and its **full Chat Completions URL**, not just the `/v1` base URL. Leave the key empty if authentication is not required.
4. Open a terminal in that directory:

```bash
# Validate the configuration and print the plan; no network requests
python3 llm_benchmark.py --dry-run

# Run the benchmark
python3 llm_benchmark.py
```

On Windows, use `python` if that is your Python command. The default configuration, report and evidence are located beside the script, even when started from another working directory. If no configuration exists, a normal first run creates a template and exits without testing; edit it and run again. `--dry-run` expects an existing configuration.

### Start with a small smoke test

Save this complete configuration as `llm_benchmark.jsonc` and replace the connection details. It schedules **one preflight and one performance request**, with warmup and self-review disabled. It checks connectivity and report generation, not capacity.

```jsonc
{
  "schema_version": "inferpulse.standalone.config/v1",
  "model": {
    "name": "Qwen3-8B", // Replace with the exact name accepted by your service
    "thinking_adapter": "auto", // Use qwen or deepseek for a deployment alias
    "api_url": "http://127.0.0.1:8000/v1/chat/completions",
    "api_key": ""
  },
  "thinking_modes": ["off"],
  "input_characters": [1024],
  "concurrency": [1],
  "output_tokens": {"off": 128, "on": 4096},
  "repetitions": 1,
  "warmup": {"enabled": false, "requests_per_length": 1},
  "timeouts": {"connect_seconds": 10, "read_seconds": 120, "total_seconds": 600},
  "self_review": false
}
```

`127.0.0.1:8000` is a placeholder. Keep `auto` when the model name is recognizable; for a deployment alias, select `qwen` or `deepseek` according to the switch syntax accepted by the service. Dry-run validates the plan, not connectivity. After the smoke test, restore a model-specific example or gradually expand the matrix.

## Supported services and thinking modes

Bench sends streaming requests to an **OpenAI-compatible Chat Completions endpoint**. By default it selects the following syntax by model name; version 1.8.0 also allows an explicit adapter:

| Adapter | Thinking off | Thinking on |
|---|---|---|
| DeepSeek | `{"thinking":{"type":"disabled"}}` | `{"thinking":{"type":"enabled"}}` |
| Qwen | `{"chat_template_kwargs":{"enable_thinking":false}}` | `{"chat_template_kwargs":{"enable_thinking":true}}` |

These are top-level JSON fields, without an `extra_body` wrapper. Matching is case-insensitive and supports organization prefixes and version suffixes, such as `deepseek-ai/DeepSeek-V3.2-Exp` and `Qwen/Qwen3-8B`. The original name is sent unchanged. If `auto` cannot recognize a name, Bench stops before any request and asks you to select an adapter.

Automatic recognition and explicit adapter selection choose request syntax; they do **not** certify deployment compatibility. Each selected mode has its own preflight. Rejected parameters are not removed and retried. Thinking observed while requesting off stops that mode. Requesting on without observable evidence is marked **unconfirmed**. HTTP 200, or the absence of visible reasoning, does not prove that a switch worked.

Set optional `model.thinking_adapter` to `auto` (default), `qwen` or `deepseek`, using lowercase values. An explicit selection takes precedence over the model name and describes the service's accepted switch format, not the model's identity. For example, a service named `production-llm` that accepts `chat_template_kwargs.enable_thinking` can use:

```jsonc
"model": {
  "name": "production-llm",
  "thinking_adapter": "qwen",
  "api_url": "http://127.0.0.1:8000/v1/chat/completions",
  "api_key": ""
}
```

The original name is sent unchanged. Preflight, warmup, performance requests and self-review use the same adapter. Bench never probes another adapter or removes rejected fields. Existing configurations that omit the option keep automatic recognition; the new field requires Bench 1.8.0 or newer. Dry-run, reports and evidence show the resolved adapter and its selection source. Offline reconstruction uses saved records, without loading current settings or recognizing the name again. Older snapshots explicitly report that the adapter and source were not recorded, while retaining their saved mode parameters.

Requests connect directly to the full HTTP(S) URL. System proxies are not used and redirects are not followed. HTTPS uses the default certificate trust store; there is no option to disable certificate verification. Sampling parameters and thinking effort use server defaults.

## Configuration

JSONC supports `//` line comments, `/* ... */` block comments and trailing commas. Comment out entries to omit them. Arrays must retain at least one entry. Only **one model object** is accepted per run.

| Setting | New template default | Supported values / meaning |
|---|---|---|
| `model.thinking_adapter` | `auto` | `auto`, `qwen`, `deepseek`; explicit selection overrides name recognition |
| `thinking_modes` | `["off", "on"]` | One or both; off always runs first |
| `input_characters` | 1024, 2048, 4096, 8192, 16384, 32768, 65536 | Unique integers from 128 to 1048576; ascending order |
| `concurrency` | `[1, 5, 10]` | Unique integers from 1 to 10, e.g. `[1, 2, 4, 6, 10]`; ascending order |
| `output_tokens` | `{"off":512,"on":4096}` | Each budget is 1–65536 tokens; keep both keys even for a single mode |
| `repetitions` | `3` | 1–1000 rounds per combination |
| `warmup` | `{"enabled":true,"requests_per_length":1}` | Serial warmup at concurrency 1 for each selected mode and length; count 1–1000 |
| `timeouts` | Connect 10 s, read idle 120 s, total 600 s | Keep all three fields; finite values greater than 0 and at most 86400 s |
| `self_review` | `true` | At most one extra request to the same model; saves its final review text |
| `seed` | Generated per run | Optional 1–64 letters, digits, underscores or hyphens for synthetic material generation; not a model sampling seed |

The 131072, 262144, 524288 and 1048576 character tiers are included but commented out in the examples. Uncomment them to opt in. **Characters are not tokens**; the service determines actual token usage and context limits.

The new template plans **42 combinations and 672 performance requests**, plus 2 preflights and 14 warmups. Self-review adds at most 1, for at most **689 requests**. Skips or cancellation reduce the actual count. Preflight uses 256 input characters and a 128-token output cap. Warmup uses the selected length and mode's full output budget. Self-review uses thinking off and a separate 2048-token budget.

In older configurations, omitted `warmup` and `self_review` stay **disabled**. Omitting tiers, concurrency, modes, budgets, rounds or timeouts restores their standard defaults. `.jsonc` takes precedence over legacy `llm_benchmark.json` when both exist. To select another file:

```bash
python3 llm_benchmark.py --config my-model.jsonc --dry-run
```

Budgets are caps, not guaranteed lengths. Reasoning and final output share the service's token accounting. Warmup may affect caching and does not establish steady-state behavior at higher concurrency. Different prefixes cannot force the server to disable caching.

## Execution and failure handling

The order is mode → input length → concurrency. Warmup precedes each length's performance combinations. Each round starts the selected number of requests together and waits for the whole batch before the next round. This is a **batch benchmark**, not a sustained arrival-rate or continuously replenished load test. Each request opens a new connection.

Preflight, warmup and self-review are excluded from performance statistics. No request is automatically retried. Ordinary warmup failures are recorded and testing continues. If all performance requests in a combination fail, higher concurrency at that length is skipped; if this occurs at concurrency 1, longer inputs in the same mode are also skipped. The other mode runs independently.

Press **Ctrl+C** to stop scheduling, close active connections and retain partial results. Abrupt termination can leave pending evidence; offline reconstruction marks incomplete work instead of treating it as success. Reconstruction does not resume testing.

## Metrics and interpretation

Timings use the client's monotonic clock and include connection, network, gateway and queueing effects. They are not pure server compute times. Material generation, serialization and evidence writes are outside the measurement window.

| Metric | Definition |
|---|---|
| TTFT, ms | Request start to first non-empty semantic text, including identifiable reasoning |
| TTFO, ms | Request start to first identifiable final-answer text |
| E2E, ms | Request start to completion, failure or timeout; distributions use successful requests |
| Per-request output tok/s | `(completion_tokens − 1) / first-to-last semantic text interval in seconds` |
| Average TPOT, ms/token | The same interval in milliseconds divided by `(completion_tokens − 1)` |
| Aggregate output tok/s | Successful output-token sum / sum of each batch's earliest-start-to-latest-end duration |
| Usage coverage | Fraction of successful requests with both input and output token counts |
| P50 / P95 | Nearest-rank quantiles over successful requests with a calculable value for that metric |

Failure time stays in the aggregate-throughput denominator. If any successful request lacks output usage, or any request is unresolved, aggregate throughput is `N/A`. Per-request speed is `N/A` with missing counts, fewer than two output tokens or an unobservable/zero text interval. SSE chunks are not tokens. Counts come **only from server usage**; `completion_tokens` is not necessarily final-answer tokens. Reasoning and cache counts are shown separately when returned.

Protocol success and final-answer presence are separate. A reasoning-only, budget-exhausted response can complete the protocol with no final answer and `TTFO=N/A`. Truncation and unknown answer status are retained. `N/A` means unavailable, not zero.

The default budgets differ: **512 off / 4096 on**. Compare actual output lengths, mode observations, execution order and sample counts. The default single-concurrency point has only three samples; its P95 has limited value. The highest successful tested concurrency is not absolute capacity. Bench does not measure answer accuracy or collect GPU/CPU telemetry, and cannot identify a hardware bottleneck from latency alone.

## HTML / PDF report preview

[Download HTML](https://gitee.com/xum1983/inferpulse-bench/blob/master/report.example.html) · [View PDF](https://gitee.com/xum1983/inferpulse-bench/blob/master/report.example.pdf)

Download the HTML file and open it directly in a browser, with no network access or dependencies. The 12-page report uses a light layout with all content expanded, paired thinking-mode comparisons, performance-change analysis, model self-review and a final appendix. Print in A4 portrait at 100% scale with browser headers and footers disabled.

![Report cover](https://gitee.com/xum1983/inferpulse-bench/raw/master/report.preview.png)

These 12-page files are a fixed design prototype using synthetic data and authored self-review text, not measurements of a real model. Version 1.9.0 generates dynamic HTML from each run with flowing tables and variable page counts; charts are not yet integrated. The dynamic renderer has A4 print styles, but long-table and long-review pagination still awaits browser visual acceptance. Prototype print checks do not validate the dynamic renderer.

## Report formats

Version 1.9.0 generates both formats by default. Comment out one entry in JSONC to disable it; retain at least one:

```jsonc
"report_formats": [
  "md",       // Comment out to disable Markdown
  "html",     // Comment out to disable HTML
],
```

Only lowercase `md` and `html` are accepted; duplicates and empty selections are rejected before requests. Omitting the entire field restores both defaults. Existing configuration files are never overwritten; add this field manually if desired and inspect `--dry-run`. Selected formats share the same timestamped basename, with copies named `report.md` / `report.html` in the evidence directory.

Offline reconstruction uses the frozen snapshot selection, not a subsequently edited configuration. Pre-1.9.0 snapshots without this field retain Markdown-only output; existing files are not deleted. HTML has no remote resources or extra dependencies, keeps all content expanded, and places self-review before the final measurement appendix. Use browser printing to save a PDF; `pdf` is not a configuration value. Formats share the same summary and do not add model requests. Adjacent-point deterioration notes include numbers and prose; the 20% screening threshold is descriptive, not a significance, SLA or capacity test.

## Reports and evidence

[View a sample report](https://gitee.com/xum1983/inferpulse-bench/blob/master/report.example.md). Its data comes from a local mock service and is not evidence of any real model's performance.

A normal run creates these files beside the script:

```text
llm_benchmark_report_<timestamp>-<id>.md      # When md is selected
llm_benchmark_report_<timestamp>-<id>.html    # When html is selected
llm_benchmark_evidence_<timestamp>-<id>/
  snapshot.json
  requests.jsonl
  summary.json
  report.md      # When md is selected
  report.html    # When html is selected
  self_review.json  # When self-review is enabled
```

The report includes the plan, separate warmup results, mode observations, latency/throughput tables, usage coverage, truncation and answer counts, and a comparison when both modes were selected. Optional self-review is model-generated commentary, not an independently validated score or cross-model ranking.

Replace `PATH_TO_EVIDENCE` with the actual directory to rebuild without network requests:

```bash
python3 llm_benchmark.py report --input PATH_TO_EVIDENCE
```

Reconstruction uses the saved snapshot and request records, not the current connection configuration. Saved self-review is reused without another model call. Keep evidence files together and retain originals for auditing.

Keys remain in plaintext in the local runtime configuration, which is ignored by Git. Keys and benchmark prompt, reasoning and answer bodies are not written to evidence. **Self-review saves its final review text.** Reports and evidence can still contain endpoints, model names, hashes and timing metadata; inspect them before sharing. Do not upload real configurations or unchecked evidence to Issues.

Since 1.7.1, reports end with a short optional [support-the-author link](https://gitee.com/xum1983/inferpulse-bench/blob/master/SPONSOR.md), after measurement notes and any model self-review. Rendering and offline reconstruction never visit the link. The note is excluded from model input and machine-readable evidence, and payment is never required to use any feature.

## Troubleshooting

| Symptom | What to check |
|---|---|
| First run creates a file and exits | Fill the generated configuration, then run again; no benchmark was sent |
| HTTP 401 / 403 | API key and service access permissions |
| Unknown model family | Keep the accepted service name and set `model.thinking_adapter` to `qwen` or `deepseek` for its switch format (1.8.0+) |
| HTTP 400 / 404 or redirect | Full path, thinking fields, context limits and output cap; fields are not dropped and redirects are not followed |
| Timeout | Distinguish connection, read-idle and total timeout; inspect network and service behavior |
| Output speed or TTFO is N/A | Check usage, multiple observable text arrivals and identifiable final-answer output |
| Unexpected thinking behavior | Read mode observations; HTTP acceptance alone is insufficient |
| Interrupted run | Rebuild a partial report from evidence; missing requests are not restarted |

`python3 llm_benchmark.py --version` prints the version; `python3 llm_benchmark.py run --help` lists options. During a benchmark, exit code `0` means the performance run completed, `2` means an error, partial run or first-time template creation, and `130` means cancellation. Check warmup and self-review separately. A successful offline rebuild exits `0` even if the historical run was incomplete.

## Feedback and license

Use [Issues](https://gitee.com/xum1983/inferpulse-bench/issues) for reproducible bugs and feature requests, and Pull Requests for focused changes. See the bilingual [contribution guide](https://gitee.com/xum1983/inferpulse-bench/blob/master/CONTRIBUTING.md).

Author: William Xu · Email: xum1983@gmail.com · License: [MIT](LICENSE) · Copyright (c) 2026 William Xu
