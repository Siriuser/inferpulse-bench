# InferPulse Bench

**Zero-dependency performance testing for self-hosted LLM services.**

[English](README.en.md) · [简体中文](README.md) · [GitHub](https://github.com/Siriuser/inferpulse-bench) · [Gitee](https://gitee.com/xum1983/inferpulse-bench)

Measure first-response latency, generation speed, concurrency and long-context behavior with a single Python file. Generate offline Markdown and HTML reports with inspectable evidence and no cloud control plane.

**Python 3.9+ · Standard-library-only runtime · MIT · Author: William Xu**

[Download 1.10.2 (GitHub)](https://github.com/Siriuser/inferpulse-bench/releases/tag/v1.10.2) · [Download 1.10.2 (Gitee)](https://gitee.com/xum1983/inferpulse-bench/releases/tag/v1.10.2) · [Usage guide (Chinese)](使用说明.md) · [Report an issue](https://github.com/Siriuser/inferpulse-bench/issues)

## What it measures

| Workload | Results |
| --- | --- |
| Baseline performance | TTFT, TTFO, request latency, per-request and aggregate tokens/s, server-reported token usage |
| Long context and concurrency | Input-character × concurrency × thinking-mode matrix, P50/P95, failures, truncation and degradation analysis |
| Thinking comparison | Parameter acceptance, observable mode evidence and off/on comparisons using Qwen- or DeepSeek-compatible parameter schemes |
| Agent performance (1.10.2) | Same-run baseline references, additional long-context cases, concurrent fixed-tool loops, per-turn tool readiness, session latency and optional PNG/WAV input |
| Reports and evidence | Markdown and HTML, A4/PDF printing, offline reconstruction and optional model-written self-review |

The Agent suite is opt-in and does not require thinking mode. Measurements and deterministic conclusions appear in a separate report section. Flow completion does not establish task correctness. Current version: 1.10.2.

## Quick start

1. Download and extract a release ZIP, or clone the repository.
2. Copy a Qwen or DeepSeek `.example.jsonc` to `llm_benchmark.jsonc`. Release ZIPs already include a blank-key default configuration.
3. Fill in the model name, full `/v1/chat/completions` endpoint and API key.
4. Run:

```bash
python3 llm_benchmark.py --dry-run
python3 llm_benchmark.py
```

On Windows, use `python` if appropriate. No package installation is required. The CLI, reports and model self-review currently use Chinese.

For a small first run, use this complete configuration and replace the model details:

```jsonc
{
  "schema_version": "inferpulse.standalone.config/v1",
  "model": {
    "name": "your-model-name",
    "thinking_adapter": "qwen",
    "api_url": "http://127.0.0.1:8000/v1/chat/completions",
    "api_key": ""
  },
  "thinking_modes": ["off"],
  "input_characters": [1024],
  "concurrency": [1],
  "repetitions": 1,
  "warmup": {"enabled": false, "requests_per_length": 1},
  "self_review": false
}
```

This schedules at most one preflight and one performance request. Set `thinking_adapter` to `qwen` or `deepseek` for the parameter scheme accepted by your service. The default, `auto`, infers the scheme from the model name; explicit selection supports deployment aliases. Preflight checks acceptance without silently dropping rejected parameters and retrying.

## Agent performance suite

Add this top-level block to your existing JSONC:

```jsonc
"agent_performance": {
  "enabled": true,
  "output_tokens": {"off": 4096, "on": 16384},
  "scenarios": ["long_context", "loop"],
  "concurrency": [1, 5, 10],
  "repetitions": 3
}
```

Defaults cover 8192 / 32768 / 65536 / 131072 input characters and fixed-tool sessions of ten model calls. Matching single-request baseline measurements are referenced within the same run; missing cases are measured separately. Loops and multimodal inputs always require dedicated measurements. With both modes and the default baseline matrix, the suite adds at most 1384 model requests including preparation. Inspect `--dry-run` and reduce the matrix for a first trial.

`loop.tool_choice` defaults to `named` (the fixed function); select `auto` explicitly when needed. There is no automatic fallback. Version 1.10.2 uses strict required arguments, explicit final-answer instructions and output-budget checks before executing a valid tool call.

Reports include per-turn data, thinking comparisons, descriptive 20% degradation flags and optional user-defined target checks. Missing usage or unresolved timing windows do not produce fabricated throughput. Tool arguments are not treated as final answers, and model-generated commands or external tools are never executed.

[Detailed Agent configuration and metrics (Chinese)](docs/AGENT_PERFORMANCE.md)

[1.10.2 changes](CHANGELOG.md) · [Report style baseline 1.0 (Chinese)](docs/design/bench-report-style/README.md)

## Reports

[HTML example](report.example.html) · [PDF example](report.example.pdf) · [Markdown example](report.example.md)

![Report preview](report.preview.png)

The examples are a 12-page design prototype using synthetic data, not model performance results. Generated report length depends on actual measurements. Version 1.10.2 leads with conclusions and charts, adds paired core-metric cards, A/B/C anomaly cards and links from chart marks to data rows, and restyles model self-review. CSS/SVG regression baselines protect the accepted design. Full A4 pagination inspection is still pending for this revision; earlier print checks do not validate this layout. Inline SVG charts and styles require no external resources. Print in A4 portrait at 100% scale with browser headers and footers disabled.

```jsonc
"report_formats": [
  "md",   // Comment out to disable Markdown
  "html", // Comment out to disable HTML
]
```

Keep at least one format. Omitting the field enables both. To rebuild offline:

```bash
python3 llm_benchmark.py report --input path/to/evidence_directory
```

Reconstruction uses the frozen run snapshot and evidence without contacting the service, reading the current configuration or requiring original media files.

## Measurement and privacy boundaries

- Input size is measured in characters, not exact tokens. Token usage comes from the service; missing values are not estimated. Client throughput is not GPU-only decoding speed.
- P95 is an empirical statistic for the available sample. The highest tested concurrency is not a capacity limit. Performance does not establish accuracy or intelligence.
- Accepted thinking parameters do not prove the requested mode took effect. Interpret output budgets, actual output lengths and observable evidence together.
- Benchmark prompts, responses, reasoning, tool payloads, media and API keys are not persisted in evidence. Enabling self-review explicitly saves its final text.
- Local configuration stores the API key in plaintext. Never publish it; review reports and metadata before sharing.

## Validation and contributions

Python 3.9 and 3.13 each pass 131 local fixture tests. On one Qwen3.6-35B-A3B / vLLM 0.25.1 deployment, post-restart named-tool tests completed 36/36 formal sessions in each condition: MTP on/32K, MTP off/8K and MTP off/32K initial characters. Tests covered both thinking modes, concurrency 1/5, three repetitions, five calls per session and unchanged 1024/4096-token budgets. Independent audits and byte-identical offline reconstruction passed. The 32K controls used identical configurations and initial-input hashes; server FSM error counts were 28/0/0. These observations do not establish the cause of earlier failures.

Earlier auto/8K and named/32K runs each completed 35/36 sessions due to one exhausted output budget; those failures remain recorded. A passing run does not guarantee sustained reliability. Subsequent fixed PNG tests completed 72/72 calls with normal termination on one deployment; this does not establish visual understanding accuracy. That deployment rejected audio input, so audio performance was not tested. Other models and services require separate verification. See the [Agent guide](docs/AGENT_PERFORMANCE.md).

```bash
python3 -m unittest discover -s tests -p 'test_standalone*benchmark.py'
```

Serving-framework feedback, redacted reproductions and documentation improvements are welcome. See [Contributing](CONTRIBUTING.md). Bench is the standalone script product; the full InferPulse desktop product has separate source, versions and licensing.

[MIT License](LICENSE) · [☕ Support the author](SPONSOR.md) · [xum1983@gmail.com](mailto:xum1983@gmail.com)

Agent calls use independent default output limits: off=4096 and on=16384 tokens per call, including reasoning. Baseline budgets remain unchanged. A higher limit does not guarantee an untruncated answer.
