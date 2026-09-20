# InferPulse Bench

**Zero-dependency performance testing for self-hosted LLM services.**

[English](README.en.md) · [简体中文](README.md) · [GitHub](https://github.com/Siriuser/inferpulse-bench) · [Gitee](https://gitee.com/xum1983/inferpulse-bench)

Measure first-response latency, generation speed, concurrency and long-context behavior with a single Python file. Generate offline Markdown and HTML reports with inspectable evidence and no cloud control plane.

**Python 3.9+ · Standard-library-only runtime · MIT · Author: William Xu**

[Download stable 1.9.0](https://github.com/Siriuser/inferpulse-bench/releases/tag/v1.9.0) · [Try 1.10.0 Agent preview](https://github.com/Siriuser/inferpulse-bench/releases/tag/v1.10.0) · [Usage guide (Chinese)](使用说明.md) · [Report an issue](https://github.com/Siriuser/inferpulse-bench/issues)

## What it measures

| Workload | Results |
| --- | --- |
| Baseline performance | TTFT, TTFO, request latency, per-request and aggregate tokens/s, server-reported token usage |
| Long context and concurrency | Input-character × concurrency × thinking-mode matrix, P50/P95, failures, truncation and degradation analysis |
| Thinking comparison | Parameter acceptance, observable mode evidence and off/on comparisons using Qwen- or DeepSeek-compatible parameter schemes |
| Agent performance (1.10.0) | Same-run baseline references, additional long-context cases, concurrent fixed-tool loops, per-turn tool readiness, session latency and optional PNG/WAV input |
| Reports and evidence | Markdown and HTML, A4/PDF printing, offline reconstruction and optional model-written self-review |

The Agent suite is opt-in and does not require thinking mode. Measurements and deterministic conclusions appear in a separate report section. Flow completion does not establish task correctness. Current source is the 1.10.0 prerelease; the stable release is 1.9.0, also available on Gitee.

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
  "scenarios": ["long_context", "loop"],
  "concurrency": [1, 5, 10],
  "repetitions": 3
}
```

Defaults cover 8192 / 32768 / 65536 / 131072 input characters and fixed-tool sessions of ten model calls. Matching single-request baseline measurements are referenced within the same run; missing cases are measured separately. Loops and multimodal inputs always require dedicated measurements. With both modes and the default baseline matrix, the suite adds at most 1084 model requests including preparation. Inspect `--dry-run` and reduce the matrix for a first trial.

Reports include per-turn data, thinking comparisons, descriptive 20% degradation flags and optional user-defined target checks. Missing usage or unresolved timing windows do not produce fabricated throughput. Tool arguments are not treated as final answers, and model-generated commands or external tools are never executed.

[Detailed Agent configuration and metrics (Chinese)](docs/AGENT_PERFORMANCE.md)

## Reports

[HTML example](report.example.html) · [PDF example](report.example.pdf) · [Markdown example](report.example.md)

![Report preview](report.preview.png)

The examples are a 12-page design prototype using synthetic data, not model performance results. Generated report length depends on actual measurements. A dynamic 1.10.0 Agent report passed Chrome A4 long-table layout inspection. Inline SVG charts and styles require no external resources. Print in A4 portrait at 100% scale with browser headers and footers disabled.

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

Python 3.9 and 3.13 each pass 112 local fixture tests covering failures, cancellation, partial evidence, report formats, tool loops, media payloads and privacy. Extracted release execution and offline reconstruction are also checked. Real-model Agent protocol and image/audio compatibility remain unverified; use actual preflight and measurements for your deployment.

```bash
python3 -m unittest discover -s tests -p 'test_standalone*benchmark.py'
```

Serving-framework feedback, redacted reproductions and documentation improvements are welcome. See [Contributing](CONTRIBUTING.md). Bench is the standalone script product; the full InferPulse desktop product has separate source, versions and licensing.

[MIT License](LICENSE) · [☕ Support the author](SPONSOR.md) · [xum1983@gmail.com](mailto:xum1983@gmail.com)
