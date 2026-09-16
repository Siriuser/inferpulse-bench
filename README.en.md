# InferPulse Bench

A standalone Python benchmark for locally deployed LLM services. Python 3.9+ standard library only; no third-party dependencies.

**Version 1.7.0 · MIT · Author: William Xu**

This repository contains the lightweight script edition, **InferPulse Bench** (`inferpulse-bench`). The full graphical product, **InferPulse** (`inferpulse`), will be published separately. Their versions, releases, configuration formats and evidence formats are separate.

## Quick start

1. Copy either model-specific `.example.jsonc` file to `llm_benchmark.jsonc` next to the script.
2. Set the model name, full Chat Completions API URL and API key.
3. Run `python3 llm_benchmark.py --dry-run` to inspect the plan, then `python3 llm_benchmark.py` to benchmark.

Supports DeepSeek/Qwen thinking-mode parameters, mode preflight, separate warmup, configurable input-character/concurrency tiers, TTFT, TTFO, E2E, output throughput, Chinese Markdown reports and offline report reconstruction. The deployment must accept the selected thinking parameters.

Input tiers count characters, not tokens. Token counts come only from server usage. Requests run in batches at concurrency 1–10; this is not a sustained-load capacity test. The default thinking/non-thinking output budgets differ. Optional model-written self-reviews are not independently verified conclusions.

Runtime configuration contains your API key and is excluded by `.gitignore`. Inspect reports and evidence before sharing them. Read the [Chinese README](README.md) and [usage guide](使用说明.md) for full details.

Email: xum1983@gmail.com · [MIT License](LICENSE) · Copyright (c) 2026 William Xu
