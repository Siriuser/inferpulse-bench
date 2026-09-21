"""Also runnable with Python 3.9 stdlib unittest; never contacts real model services."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "llm_benchmark.py"
SPEC = importlib.util.spec_from_file_location("standalone_benchmark", SCRIPT)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)
SECRET = "test-key-never-persist-this"
ANSWER = "private-response-never-persist-this"
THOUGHT = "private-thinking-never-persist-this"


def config_for(url="http://127.0.0.1:1/v1/chat/completions", name="Qwen3.8-27B"):
    return bench.validate_config(
        {
            "schema_version": bench.CONFIG_VERSION,
            "model": {"name": name, "api_url": url, "api_key": SECRET},
            "input_characters": [128, 512],
            "concurrency": [1, 5, 10],
            "repetitions": 1,
            "timeouts": {"connect_seconds": 1, "read_seconds": 2, "total_seconds": 4},
            "seed": "test-seed",
        }
    )


def enabled(body):
    if "thinking" in body:
        return body["thinking"]["type"] == "enabled"
    return body["chat_template_kwargs"]["enable_thinking"]


def event(handler, value):
    raw = "data: " + (value if isinstance(value, str) else json.dumps(value)) + "\n\n"
    handler.wfile.write(raw.encode())
    handler.wfile.flush()


def start_stream(handler):
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Connection", "close")
    handler.end_headers()


def completion(handler, body, usage=True, thinking=None):
    start_stream(handler)
    event(handler, {"choices": [{"delta": {"role": "assistant"}}]})
    if enabled(body) if thinking is None else thinking:
        event(handler, {"choices": [{"delta": {"reasoning_content": THOUGHT}}]})
        time.sleep(0.005)
    event(handler, {"choices": [{"delta": {"content": ANSWER}}]})
    time.sleep(0.01)
    event(handler, {"choices": [{"delta": {"content": "second-part"}}]})
    last = {"choices": [{"delta": {}, "finish_reason": "length"}]}
    if usage:
        last["usage"] = {
            "prompt_tokens": 100,
            "completion_tokens": 12,
            "completion_tokens_details": {"reasoning_tokens": 3 if enabled(body) else 0},
        }
    event(handler, last)
    event(handler, "[DONE]")


REVIEW_TEXT = (
    "综合档位：人上人\n1. 本次数据反映已测场景的表现。\n2. 不能把最高已测并发当作容量上限。"
)


def is_review(body):
    return body["messages"][0]["role"] == "system"


def is_warmup(body):
    text = body["messages"][0]["content"]
    mode = "on" if enabled(body) else "off"
    return any(
        text.startswith(
            hashlib.sha256(
                f"test-seed:warmup-{mode}-{len(text)}-r{number}-q1".encode()
            ).hexdigest()[:32]
            + "\n"
        )
        for number in range(1, 4)
    )


def review_response(handler, text=REVIEW_TEXT, reason="stop"):
    start_stream(handler)
    event(handler, {"choices": [{"delta": {"content": text[:7]}}]})
    event(
        handler,
        {
            "choices": [{"delta": {"content": text[7:]}, "finish_reason": reason}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 40},
        },
    )
    event(handler, "[DONE]")


class LocalServer:
    def __init__(self, behavior=completion):
        self.behavior = behavior
        self.requests = []
        self.active = 0
        self.peaks = {}
        self.lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                group = (enabled(body), len(body["messages"][0]["content"]))
                with owner.lock:
                    owner.requests.append((body, dict(self.headers)))
                    owner.active += 1
                    owner.peaks[group] = max(owner.peaks.get(group, 0), owner.active)
                try:
                    owner.behavior(self, body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    with owner.lock:
                        owner.active -= 1

        class Server(ThreadingHTTPServer):
            request_queue_size = 128

        self.httpd = Server(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(
            target=lambda: self.httpd.serve_forever(poll_interval=0.01), daemon=True
        )
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1/chat/completions"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)


def single(config, mode="off", controller=None):
    gate = threading.Event()
    gate.set()
    spec = {
        "request_id": "sample",
        "batch_id": "batch",
        "cell_id": "cell",
        "phase": "performance",
        "mode": mode,
    }
    body = json.dumps(
        {
            "model": config["model"]["name"],
            "messages": [{"role": "user", "content": "probe"}],
            **bench.request_mode_parameters(config, mode),
        }
    ).encode()
    return bench.perform_request(
        config, spec, body, SECRET, controller or bench.StopController(), gate, time.perf_counter()
    )


class ConfigurationTests(unittest.TestCase):
    def test_thinking_adapter_defaults_and_explicit_override(self):
        for name, requested, resolved, source in (
            ("org/QWEN3-8B", None, "qwen", "model_name"),
            ("DeepSeek-V3", "auto", "deepseek", "model_name"),
            ("production-llm", "qwen", "qwen", "explicit"),
            ("内网/服务-A", "deepseek", "deepseek", "explicit"),
            ("DeepSeek-V3", "qwen", "qwen", "explicit"),
            ("Qwen3-8B", "deepseek", "deepseek", "explicit"),
        ):
            with self.subTest(name=name, requested=requested):
                raw = config_for()
                raw["model"].update(name=name, api_key=SECRET)
                if requested is None:
                    raw["model"].pop("thinking_adapter")
                else:
                    raw["model"]["thinking_adapter"] = requested
                config = bench.validate_config(raw)
                self.assertEqual(config["model"]["name"], name)
                self.assertEqual(config["model"]["thinking_adapter"], requested or "auto")
                self.assertEqual(
                    bench.resolve_thinking_adapter(config["model"]),
                    {"requested": requested or "auto", "resolved": resolved, "source": source},
                )
                self.assertNotIn(SECRET, json.dumps(config))

    def test_invalid_adapters_and_names_are_rejected_without_leaking_values(self):
        for adapter in (None, "", "QWEN", "qwen ", SECRET, True, 1, [], {}):
            config = config_for()
            config["model"].update(name="production-llm", thinking_adapter=adapter)
            with self.subTest(adapter=adapter), self.assertRaises(bench.BenchmarkError) as caught:
                bench.validate_config(config)
            self.assertIn("model.thinking_adapter", str(caught.exception))
            self.assertNotIn(SECRET, str(caught.exception))
        for name in (None, [], 1, "", "   ", "alias\n", "a\x00b", "a" * 201):
            config = config_for()
            config["model"].update(name=name, thinking_adapter="qwen")
            with self.subTest(name=name), self.assertRaises(bench.BenchmarkError):
                bench.validate_config(config)

    def test_alias_dry_run_and_template_are_offline_and_redacted(self):
        config = config_for()
        config["model"].update(name="production-llm", thinking_adapter="qwen", api_key=SECRET)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.jsonc"
            path.write_text(bench.config_template(config))
            stdout = io.StringIO()
            with (
                mock.patch.object(bench.socket, "create_connection", side_effect=AssertionError),
                contextlib.redirect_stdout(stdout),
            ):
                self.assertEqual(bench.main(["--config", str(path), "--dry-run"]), 0)
            plan = json.loads(stdout.getvalue())
            self.assertEqual(plan["model"], "production-llm")
            self.assertEqual(
                plan["thinking_adapter"],
                {"requested": "qwen", "resolved": "qwen", "source": "explicit"},
            )
            self.assertEqual(
                plan["mode_parameters"],
                {
                    "off": {"chat_template_kwargs": {"enable_thinking": False}},
                    "on": {"chat_template_kwargs": {"enable_thinking": True}},
                },
            )
            self.assertNotIn(SECRET, stdout.getvalue())
            self.assertEqual(list(Path(temporary).iterdir()), [path])
            self.assertEqual(bench.read_config(path)["model"], config["model"])

    def test_model_only_configuration_expands_defaults_without_retaining_key(self):
        raw = {
            "schema_version": bench.CONFIG_VERSION,
            "model": {
                "name": "Qwen3.8-27B",
                "api_url": "http://127.0.0.1:1/v1/chat/completions",
                "api_key": SECRET,
            },
        }
        config = bench.validate_config(raw)
        self.assertNotIn("api_key", config["model"])
        self.assertNotIn(SECRET, json.dumps(config))
        self.assertEqual(bench.read_credential(raw), SECRET)
        self.assertEqual(config["input_characters"], bench.DEFAULT_CHARACTERS)
        self.assertEqual(config["repetitions"], 3)
        for key in (None, "", "\r\n" + SECRET):
            with self.assertRaises(bench.BenchmarkError) as caught:
                bench.read_credential({"model": {"api_key": key}})
            self.assertNotIn(SECRET, str(caught.exception))

    def test_defaults_and_both_examples(self):
        for name in ("deepseek", "qwen"):
            raw = bench.read_config(ROOT / f"llm_benchmark.{name}.example.jsonc")
            self.assertEqual(raw["output_tokens"], {"off": 512, "on": 4096})
            self.assertEqual(raw["repetitions"], 3)
            self.assertEqual(
                raw["timeouts"], {"connect_seconds": 10, "read_seconds": 120, "total_seconds": 600}
            )
            config = bench.validate_config(raw)
            cells = bench.make_cells(config)
            self.assertEqual(len(cells), 42)
            self.assertEqual(sum(cell["concurrency"] * cell["repetitions"] for cell in cells), 672)
            self.assertEqual(config["input_characters"], bench.DEFAULT_CHARACTERS)
            self.assertEqual(max(config["input_characters"]), 65536)
            self.assertEqual(config["concurrency"], [1, 5, 10])
            self.assertEqual(config["thinking_modes"], ["off", "on"])
            self.assertEqual(config["output_tokens"], {"off": 512, "on": 4096})
            self.assertEqual(config["warmup"], {"enabled": True, "requests_per_length": 1})
            self.assertEqual(len(bench.make_warmups(config)), 14)

    def test_jsonc_comments_preserve_strings_credentials_and_advanced_values(self):
        raw = config_for()
        raw["model"]["api_key"] = SECRET + r'//path/*comment*/#,]"escaped\\'
        source = "\ufeff/* configuration */\n" + bench.config_template(raw) + "// end"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.jsonc"
            path.write_text(source, encoding="utf-8")
            self.assertEqual(bench.read_config(path), raw)
            with self.assertRaises(bench.BenchmarkError):
                bench.read_json(path)  # Evidence JSON is not relaxed by this feature.
            normalized = bench.validate_config(bench.read_config(path))
            self.assertNotIn(SECRET, json.dumps(normalized))

    def test_commenting_each_ladder_entry_removes_only_that_entry(self):
        source = bench.config_template()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.jsonc"
            for key, entries in (
                ("input_characters", bench.DEFAULT_CHARACTERS),
                ("concurrency", [1, 5, 10]),
                ("thinking_modes", list(bench.MODES)),
            ):
                for entry in entries:
                    with self.subTest(key=key, entry=entry):
                        line = "    " + json.dumps(entry) + ","
                        path.write_text(source.replace(line, "    // " + line.strip()))
                        config = bench.validate_config(bench.read_config(path))
                        self.assertEqual(config[key], [item for item in entries if item != entry])
                commented = source
                for entry in entries:
                    line = "    " + json.dumps(entry) + ","
                    commented = commented.replace(line, "    // " + line.strip())
                path.write_text(commented)
                with self.assertRaises(bench.BenchmarkError):
                    bench.validate_config(bench.read_config(path))

    def test_jsonc_invalid_syntax_has_safe_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.jsonc"
            for text in (
                '{"key": "' + SECRET,
                '{"key": 1 /* ' + SECRET,
                '{"key": 1,,}',
                '{"key": 1 /*' + SECRET + "*/ 2}",
                '# YAML is not JSONC\nkey: "' + SECRET + '"',
            ):
                path.write_text(text)
                with self.assertRaises(bench.BenchmarkError) as caught:
                    bench.read_config(path)
                self.assertNotIn(SECRET, str(caught.exception))

    def test_mode_selection_validation_and_canonical_execution_order(self):
        for selection, expected in (
            (["on"], ["on"]),
            (["off"], ["off"]),
            (["on", "off"], ["off", "on"]),
        ):
            config = bench.validate_config(dict(config_for(), thinking_modes=selection))
            self.assertEqual(config["thinking_modes"], expected)
            modes = list(dict.fromkeys(cell["mode"] for cell in bench.make_cells(config)))
            self.assertEqual(modes, expected)
        for selection in ([], "off", None, [True], ["off", "off"], ["auto"], [{}], ["ON"]):
            with self.subTest(selection=selection), self.assertRaises(bench.BenchmarkError):
                bench.validate_config(dict(config_for(), thinking_modes=selection))

    def test_commented_plan_dry_run_is_isolated_and_needs_no_key(self):
        raw = {
            "model": dict(bench.DEFAULT_MODEL, api_key=SECRET),
            "repetitions": 2,
            "timeouts": {"connect_seconds": 7, "read_seconds": 31, "total_seconds": 91},
        }
        source = bench.config_template(raw)
        for value in [2048, 4096, 8192, 16384, 32768, 65536, 10, "on"]:
            line = "    " + json.dumps(value) + ","
            source = source.replace(line, "    // " + line.strip())
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            script = directory / "llm_benchmark.py"
            shutil.copyfile(SCRIPT, script)
            config = directory / bench.DEFAULT_CONFIG_NAME
            config.write_text(source)
            result = subprocess.run(
                [sys.executable, "-I", "-S", str(script), "--dry-run"],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn(SECRET, result.stdout + result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["input_characters"], [1024])
            self.assertEqual(plan["concurrency"], [1, 5])
            self.assertEqual(plan["thinking_modes"], ["off"])
            self.assertEqual(plan["cells"], 2)
            self.assertEqual(plan["performance_requests"], 12)
            self.assertEqual(plan["repetitions"], 2)
            self.assertEqual(plan["timeouts"], raw["timeouts"])
            self.assertEqual(plan["preflight_requests"], 1)
            self.assertEqual(set(plan["mode_parameters"]), {"off"})
            self.assertEqual(set(directory.iterdir()), {script, config})

    def test_model_matching_and_parameters(self):
        for name in (
            "DeepSeek",
            "deepseek-chat",
            "deepseek-reasoner",
            "DeepSeek-R1",
            "deepseek-ai/DeepSeek-V3.2-Exp",
            "DeepSeek-V4-Flash-0731",
            "org/DEEPSEEK-V4-FLASH-fp8",
            "org/DeepSeek-V99-Custom-AWQ",
        ):
            self.assertEqual(bench.model_family(name), "deepseek")
        for name in (
            "Qwen",
            "Qwen2.5-7B-Instruct",
            "Qwen/Qwen3-8B",
            "org/QWEN3-235B-A22B-Instruct-FP8",
            "org/qwen99-Custom-AWQ",
            "Qwen3.8-27B",
            "Qwen/qwen3.8-27b-FP8",
            "Qwen3.6-35B-A3B",
            "Qwen/qwen3.6-35b-a3b-FP8",
        ):
            self.assertEqual(bench.model_family(name), "qwen")
        for name in (
            "mystery",
            "Llama-3.3-70B",
            "Qwenish-8B",
            "DeepSeeker-V3",
            "not-qwen3-8b",
            "Qwen/unknown",
            "DeepSeek/",
            "Qwen3-8B\n",
            "",
            "Qwen" + "3" * 200,
            [],
            None,
        ):
            with self.subTest(name=name), self.assertRaises(bench.BenchmarkError):
                bench.model_family(name)
        self.assertEqual(
            bench.mode_parameters("deepseek", "off"), {"thinking": {"type": "disabled"}}
        )
        self.assertEqual(
            bench.mode_parameters("qwen", "on"), {"chat_template_kwargs": {"enable_thinking": True}}
        )
        self.assertEqual(bench.mode_parameters("deepseek", "on"), {"thinking": {"type": "enabled"}})
        self.assertEqual(
            bench.mode_parameters("qwen", "off"),
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    def test_rejects_multiple_models_and_invalid_values(self):
        config = config_for()
        invalid = [
            dict(config, model=[config["model"]]),
            dict(config, models=[config["model"]]),
            dict(config, input_characters=[1048577]),
            dict(config, concurrency=[11]),
            dict(config, concurrency=[True]),
            dict(config, concurrency=[1, 1]),
            dict(config, repetitions=0),
            dict(config, seed="bad seed"),
            dict(config, output_tokens={"off": 512}),
            dict(config, output_tokens={"off": 0, "on": 4096}),
            dict(config, output_tokens={"off": 512, "on": 65537}),
            dict(config, output_tokens={"off": True, "on": 4096}),
            dict(config, output_tokens={"off": 512, "on": "8192"}),
            dict(config, schema_version="old"),
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(bench.BenchmarkError):
                bench.validate_config(value)
        for url in ("http://user:secret@host/v1", "http://host/v1?key=secret", "file:///tmp/a"):
            value = dict(config, model=dict(config["model"], api_url=url))
            with self.assertRaises(bench.BenchmarkError):
                bench.validate_config(value)
        for value in (float("nan"), float("inf"), 0, True):
            with self.assertRaises(bench.BenchmarkError):
                bench.validate_config(
                    dict(
                        config,
                        timeouts={"connect_seconds": value, "read_seconds": 1, "total_seconds": 1},
                    )
                )

    def test_prompt_lengths_hashes_and_unique_prefixes(self):
        for size in bench.DEFAULT_CHARACTERS + bench.OPTIONAL_CHARACTERS:
            prompt = bench.make_prompt(size, "seed", "request")
            self.assertEqual(len(prompt), size)
            self.assertEqual(prompt, bench.make_prompt(size, "seed", "request"))
            self.assertNotEqual(prompt[:32], bench.make_prompt(size, "seed", "other")[:32])

    def test_extended_template_levels_are_opt_in_and_support_one_million_characters(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.jsonc"
            for family in ("deepseek", "qwen"):
                example = ROOT / f"llm_benchmark.{family}.example.jsonc"
                source = example.read_text(encoding="utf-8")
                self.assertEqual(
                    bench.read_config(example)["input_characters"], bench.DEFAULT_CHARACTERS
                )
                for size in bench.OPTIONAL_CHARACTERS:
                    self.assertIn(f"    // {size},", source)
                    path.write_text(source.replace(f"    // {size},", f"    {size},"))
                    config = bench.validate_config(bench.read_config(path))
                    self.assertEqual(config["input_characters"], bench.DEFAULT_CHARACTERS + [size])
                    self.assertEqual(len(bench.make_cells(config)), 48)
                    self.assertEqual(
                        max(cell["input_characters"] for cell in bench.make_cells(config)), size
                    )
                all_enabled = source
                for size in bench.OPTIONAL_CHARACTERS:
                    all_enabled = all_enabled.replace(f"    // {size},", f"    {size},")
                path.write_text(all_enabled)
                config = bench.validate_config(bench.read_config(path))
                cells = bench.make_cells(config)
                self.assertEqual(len(cells), 66)
                self.assertEqual(
                    sum(cell["concurrency"] * cell["repetitions"] for cell in cells), 1056
                )
                self.assertEqual(max(config["input_characters"]), 1048576)
                path.write_text(bench.config_template(config))
                self.assertEqual(
                    bench.read_config(path)["input_characters"], config["input_characters"]
                )

    def test_isolated_dry_run_needs_no_key_or_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            shutil.copyfile(SCRIPT, directory / "benchmark.py")
            shutil.copyfile(
                ROOT / "llm_benchmark.qwen.example.jsonc", directory / "model.json"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "benchmark.py",
                    "run",
                    "--config",
                    "model.json",
                    "--dry-run",
                ],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["performance_requests"], 672)
            self.assertEqual(
                json.loads(result.stdout)["thinking_adapter"],
                {"requested": "auto", "resolved": "qwen", "source": "model_name"},
            )
            self.assertFalse((directory / "evidence").exists())


class ParsingTests(unittest.TestCase):
    def test_sse_byte_splits_unicode_crlf_multiline_and_comments(self):
        decoder = bench.SSEDecoder()
        raw = ': 心跳\r\ndata: {"choices":\r\ndata: []}\r\n\r\ndata: [DONE]\r\r'.encode()
        events = []
        for byte in raw:
            events.extend(decoder.feed(bytes([byte])))
        self.assertEqual(events, ['{"choices":\n[]}', "[DONE]"])

    def test_split_think_tags_keep_original_arrival_times(self):
        observation = bench.TextObservation("on")
        observation.add({"content": "<thi"}, 0.1)
        observation.add({"content": "nk>推理"}, 0.2)
        observation.add({"content": "</thi"}, 0.3)
        observation.add({"content": "nk>答案"}, 0.4)
        result = observation.summarize()
        self.assertEqual(result["first_text"], 0.2)
        self.assertEqual(result["first_answer"], 0.4)
        self.assertTrue(result["reasoning_observed"])
        self.assertNotIn("推理", json.dumps(result))

    def test_empty_think_and_prefilled_opening_tag(self):
        for text, reasoning in (
            ("<think> \n </think>answer", False),
            ("reason</think>answer", True),
        ):
            observation = bench.TextObservation("off")
            observation.add({"content": text}, 0.2)
            result = observation.summarize()
            self.assertEqual(result["reasoning_observed"], reasoning)
            self.assertTrue(result["answer_present"])

    def test_reasoning_only_and_ambiguous_content(self):
        for delta, observed, answer in (
            ({"reasoning_content": THOUGHT}, True, False),
            ({"content": "<think>still thinking"}, True, False),
            ({"content": "unclassified"}, False, None),
        ):
            observation = bench.TextObservation("on")
            observation.add(delta, 0.2)
            result = observation.summarize()
            self.assertEqual(result["reasoning_observed"], observed)
            self.assertIs(result["answer_present"], answer)
            self.assertIsNone(result["first_answer"])
            self.assertEqual(result["first_text"], 0.2)

    def test_oversize_event_is_rejected(self):
        decoder = bench.SSEDecoder()
        with self.assertRaises(bench.BenchmarkError):
            list(decoder.feed(b"x" * (bench.MAX_EVENT_CHARACTERS + 1)))


class RequestTests(unittest.TestCase):
    def test_reasoning_and_visible_timing_with_tail_usage(self):
        with LocalServer() as server:
            result = single(config_for(server.url), mode="on")
        self.assertEqual(result["status"], "success")
        self.assertLess(result["timings_ms"]["ttft"], result["timings_ms"]["ttfo"])
        self.assertEqual(result["usage"]["reasoning_tokens"], 3)
        self.assertTrue(result["truncated"])
        for secret in (SECRET, ANSWER, THOUGHT):
            self.assertNotIn(secret, json.dumps(result))

    def test_usage_absence_and_single_fragment_are_not_fabricated(self):
        def behavior(handler, body):
            start_stream(handler)
            event(handler, {"choices": [{"delta": {"content": ANSWER}, "finish_reason": "stop"}]})
            event(handler, "[DONE]")

        with LocalServer(behavior) as server:
            result = single(config_for(server.url))
        metrics = bench.calculate_metrics([result])
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["timings_ms"]["output_duration"], 0)
        self.assertEqual(metrics["usage_coverage"], 0)
        self.assertIsNone(metrics["output_tps"]["p50"])
        self.assertIsNone(metrics["aggregate_output_tps"])

    def test_only_reasoning_with_length_stop_is_protocol_success(self):
        def behavior(handler, body):
            start_stream(handler)
            event(
                handler,
                {
                    "choices": [
                        {"delta": {"reasoning_content": THOUGHT}, "finish_reason": "length"}
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4096},
                },
            )
            event(handler, "[DONE]")

        with LocalServer(behavior) as server:
            result = single(config_for(server.url), mode="on")
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["truncated"])
        self.assertFalse(result["answer_present"])
        self.assertIsNone(result["timings_ms"]["ttfo"])

    def test_http_rejection_never_follows_redirect_or_logs_body(self):
        for status in (302, 400, 401, 403, 429, 500):

            def behavior(handler, body, code=status):
                handler.send_response(code)
                handler.send_header("Location", "http://127.0.0.1:1/leak")
                handler.end_headers()
                handler.wfile.write(SECRET.encode())

            with self.subTest(status=status), LocalServer(behavior) as server:
                result = single(config_for(server.url))
                self.assertEqual(len(server.requests), 1)
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["http_status"], status)
                self.assertNotIn(SECRET, json.dumps(result))

    def test_bad_sse_empty_and_interrupted_responses(self):
        cases = [
            (["not json " + SECRET], "protocol_error"),
            ([{"choices": [{"delta": {"role": "assistant"}}]}, "[DONE]"], "empty_response"),
            ([{"choices": [{"delta": {"content": ANSWER}}]}], "stream_interrupted"),
            ([{"error": {"message": SECRET}}], "api_error"),
            ([{"choices": "not a list"}], "protocol_error"),
        ]
        for events, error in cases:

            def behavior(handler, body, records=events):
                start_stream(handler)
                for value in records:
                    event(handler, value)

            with self.subTest(error=error), LocalServer(behavior) as server:
                result = single(config_for(server.url))
                self.assertEqual(result["error"], error)
                self.assertNotIn(SECRET, json.dumps(result))

    def test_idle_and_total_deadlines(self):
        for trickle in (False, True):

            def behavior(handler, body, send=trickle):
                start_stream(handler)
                event(handler, {"choices": [{"delta": {"content": "partial"}}]})
                for _ in range(40):
                    time.sleep(0.02)
                    if send:
                        handler.wfile.write(b": heartbeat\n\n")
                        handler.wfile.flush()

            with self.subTest(trickle=trickle), LocalServer(behavior) as server:
                config = config_for(server.url)
                config["timeouts"] = {
                    "connect_seconds": 0.2,
                    "read_seconds": 0.1,
                    "total_seconds": 0.25,
                }
                started = time.perf_counter()
                result = single(config)
                self.assertLess(time.perf_counter() - started, 0.7)
                self.assertEqual(result["status"], "timeout")
                self.assertIsNotNone(result["timings_ms"]["ttft"])


class MetricsTests(unittest.TestCase):
    def test_invalid_usage_never_becomes_a_token_measurement(self):
        for value in (-1, True, 1.5, "12", 10**400, float("inf")):
            usage = {}
            bench.usage_from({"usage": {"prompt_tokens": value, "completion_tokens": value}}, usage)
            self.assertEqual(usage, {})

    def sample(self, request_id, batch, start, end, tokens=5, status="success"):
        return {
            "request_id": request_id,
            "batch_id": batch,
            "attempted": True,
            "status": status,
            "error": None if status == "success" else "timeout",
            "start_offset_s": start,
            "end_offset_s": end,
            "timings_ms": {
                "ttft": 100,
                "ttfo": 200,
                "e2e": (end - start) * 1000,
                "output_duration": 2000,
            },
            "usage": {"prompt_tokens": 20, "completion_tokens": tokens},
            "answer_present": True,
            "reasoning_observed": False,
        }

    def test_formulas_use_batch_wall_time_and_include_failures(self):
        requests = [
            self.sample("a", "b1", 1, 4),
            self.sample("b", "b1", 1.1, 5),
            self.sample("c", "b2", 10, 12, status="timeout"),
        ]
        m = bench.calculate_metrics(requests)
        self.assertEqual(m["duration_seconds"], 6)
        self.assertEqual(m["output_tps"]["p50"], 2)
        self.assertEqual(m["tpot_ms"]["p50"], 500)
        self.assertEqual(m["aggregate_output_tps"], 10 / 6)
        self.assertEqual(m["success_rate"], 2 / 3)
        self.assertEqual(m["latency_ms"]["e2e"]["count"], 2)
        self.assertEqual(bench.distribution([10, 20, 30])["p95"], 30)

    def test_partial_usage_and_unresolved_requests(self):
        requests = [self.sample("a", "b1", 0, 3), self.sample("b", "b1", 0, 3)]
        requests[1]["usage"].pop("completion_tokens")
        m = bench.calculate_metrics(requests)
        self.assertIsNone(m["aggregate_output_tps"])
        self.assertEqual(m["output_tps"]["count"], 1)
        self.assertEqual(m["usage_coverage"], 0.5)
        m = bench.calculate_metrics(requests, unresolved=1)
        self.assertIsNone(m["success_rate"])
        self.assertIsNone(m["duration_seconds"])


class RunnerTests(unittest.TestCase):
    def test_explicit_adapter_keeps_preflight_rejection_and_observation_rules(self):
        def configure(config):
            config["model"].update(name="production-llm", thinking_adapter="qwen")
            config.update(input_characters=[128], concurrency=[1])

        for behavior_name, expected_requests in (
            ("rejected", 2),
            ("ignored", 4),
            ("contradicted", 3),
        ):

            def behavior(handler, body, behavior_name=behavior_name):
                if behavior_name == "rejected":
                    handler.send_response(400)
                    handler.end_headers()
                else:
                    completion(handler, body, usage=False, thinking=behavior_name == "contradicted")

            with self.subTest(behavior=behavior_name):
                summary, _, _, requests, _, _ = self.run_local(behavior, configure)
                self.assertEqual(len(requests), expected_requests)
                for body, _ in requests:
                    self.assertIn("chat_template_kwargs", body)
                    self.assertNotIn("thinking", body)
                if behavior_name == "rejected":
                    self.assertTrue(
                        all(not mode["parameters_accepted"] for mode in summary["modes"].values())
                    )
                    self.assertEqual(summary["status"], "partial")
                elif behavior_name == "ignored":
                    self.assertEqual(summary["modes"]["on"]["observation"], "unconfirmed")
                    self.assertEqual(summary["modes"]["off"]["observation"], "not_observed")
                else:
                    self.assertEqual(summary["modes"]["off"]["observation"], "contradicted")
                    self.assertEqual(summary["modes"]["on"]["observation"], "observed")

    def run_local(self, behavior=completion, configure=None):
        with tempfile.TemporaryDirectory() as temporary, LocalServer(behavior) as server:
            config = config_for(server.url)
            if configure:
                configure(config)
            directory = Path(temporary) / "run"
            with contextlib.redirect_stdout(io.StringIO()):
                summary = bench.run_benchmark(config, directory, SECRET)
            snapshot, records, warnings = bench.load_evidence(directory)
            reconstructed = bench.generate_report(directory)
            self.assertEqual(summary, reconstructed)
            self.assertFalse(warnings)
            report = (directory / "report.md").read_text()
            persisted = "\n".join(path.read_text() for path in directory.iterdir())
            for secret in (SECRET, ANSWER, THOUGHT):
                self.assertNotIn(secret, persisted)
            for body, headers in server.requests:
                self.assertEqual(headers["Authorization"], "Bearer " + SECRET)
                self.assertNotIn("extra_body", body)
                self.assertEqual(body["model"], config["model"]["name"])
                self.assertNotIn(body["messages"][0]["content"], persisted)
            return (
                summary,
                records,
                report,
                copy.deepcopy(server.requests),
                dict(server.peaks),
                snapshot,
            )

    def test_one_million_character_input_reaches_service_and_rebuilt_report(self):
        def configure(config):
            config.update(input_characters=[1048576], concurrency=[1], thinking_modes=["off"])

        summary, records, report, requests, _, snapshot = self.run_local(configure=configure)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(
            [len(body["messages"][0]["content"]) for body, _ in requests], [256, 1048576]
        )
        self.assertEqual(snapshot["config"]["input_characters"], [1048576])
        self.assertEqual(summary["planned_performance_requests"], 1)
        performance = [
            row["request"]
            for row in records
            if row["record_type"] == "result" and row["request"]["phase"] == "performance"
        ]
        self.assertEqual(performance[0]["input_characters"], 1048576)
        self.assertIn("最高 1048576 字符", report)

    def test_custom_output_budgets_reach_requests_evidence_and_rebuilt_report(self):
        budgets = {"off": 768, "on": 8192}

        def configure(config):
            config.update(input_characters=[128], concurrency=[1], output_tokens=budgets)

        summary, records, report, requests, _, snapshot = self.run_local(configure=configure)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["config"]["output_tokens"], budgets)
        self.assertEqual(snapshot["config"]["output_tokens"], budgets)
        self.assertEqual(len(requests), 4)
        for body, _ in requests:
            mode = "on" if enabled(body) else "off"
            is_probe = len(body["messages"][0]["content"]) == 256
            self.assertEqual(body["max_tokens"], 128 if is_probe else budgets[mode])
        for record in records:
            if record["record_type"] in {"scheduled", "result"}:
                request = record["request"]
                if request["phase"] == "performance":
                    self.assertEqual(request["max_tokens"], budgets[request["mode"]])
        self.assertIn("关闭思考 768 Token；开启思考 8192 Token", report)

    def test_single_mode_selected_ladders_request_bodies_and_report_reconstruction(self):
        for mode in bench.MODES:
            with self.subTest(mode=mode):

                def configure(config, selected=mode):
                    config.update(thinking_modes=[selected], concurrency=[5])

                summary, records, report, requests, _, snapshot = self.run_local(
                    configure=configure
                )
                self.assertEqual(summary["status"], "completed")
                self.assertEqual(summary["planned_cells"], 2)
                self.assertEqual(summary["planned_performance_requests"], 10)
                self.assertEqual(summary["planned_preflight_requests"], 1)
                self.assertEqual(len(requests), 11)
                self.assertEqual(set(summary["modes"]), {mode})
                self.assertEqual(set(snapshot["mode_parameters"]), {mode})
                self.assertEqual(snapshot["config"]["thinking_modes"], [mode])
                results = [r["request"] for r in records if r["record_type"] == "result"]
                self.assertEqual(sum(r["phase"] == "preflight" for r in results), 1)
                self.assertTrue(all(enabled(body) == (mode == "on") for body, _ in requests))
                self.assertNotIn("## 双模式对照", report)
                self.assertIn("另有 1 次独立模式预检", report)
                self.assertIn("本次只选择一种思考模式", report)
                excluded = "off" if mode == "on" else "on"
                self.assertNotIn("## " + bench.MODE_LABELS[excluded], report)
                self.assertNotIn(bench.MODE_LABELS[excluded] + " →", report)

    def test_selected_lowest_concurrency_failure_skips_larger_inputs(self):
        def behavior(handler, body):
            if len(body["messages"][0]["content"]) == 256:
                completion(handler, body)
            else:
                handler.send_response(500)
                handler.end_headers()

        def configure(config):
            config.update(thinking_modes=["on"], concurrency=[5, 10])

        summary, _, _, requests, _, _ = self.run_local(behavior, configure)
        self.assertEqual(len(requests), 6)
        self.assertEqual(summary["status"], "partial")
        self.assertEqual(summary["modes"]["on"]["stop_reason"], "lowest_concurrency_all_failed")
        self.assertTrue(all(row["status"] == "skipped" for row in summary["cells"][1:]))

    def test_matrix_separates_modes_usage_and_request_counts(self):
        summary, records, report, requests, peaks, snapshot = self.run_local()
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(len(requests), 66)  # 2 lengths * (1+5+10) * 2 modes + 2 probes.
        results = [row["request"] for row in records if row["record_type"] == "result"]
        performance = [row for row in results if row["phase"] == "performance"]
        self.assertEqual(len(performance), 64)
        self.assertEqual(sum(row["metrics"]["attempted_count"] for row in summary["cells"]), 64)
        self.assertEqual([enabled(body) for body, _ in requests], [False] * 33 + [True] * 33)
        self.assertEqual(summary["modes"]["off"]["observation"], "not_observed")
        self.assertEqual(summary["modes"]["on"]["observation"], "observed")
        for row in performance:
            self.assertEqual(row["max_tokens"], 4096 if row["mode"] == "on" else 512)
        self.assertLessEqual(max(peaks.values()), 10)
        self.assertEqual(snapshot["config"]["seed"], "test-seed")
        self.assertIn("双模式对照", report)
        self.assertIn("最高 512 字符", report)
        self.assertIn("不同输出预算", report)

    def test_full_default_plan_executes_672_performance_14_warmups_and_two_probes(self):
        def configure(config):
            config.update(
                input_characters=bench.DEFAULT_CHARACTERS,
                repetitions=3,
                warmup={"enabled": True, "requests_per_length": 1},
            )

        summary, records, _, requests, _, _ = self.run_local(configure=configure)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(len(summary["cells"]), 42)
        self.assertEqual(len(requests), 688)
        self.assertEqual(summary["warmup"]["planned_requests"], 14)
        self.assertEqual(summary["warmup"]["metrics"]["success_count"], 14)
        self.assertEqual(sum(row["metrics"]["attempted_count"] for row in summary["cells"]), 672)
        self.assertEqual(max(row["input_characters"] for row in summary["cells"]), 65536)
        results = [record["request"] for record in records if record["record_type"] == "result"]
        self.assertEqual(sum(row["phase"] == "preflight" for row in results), 2)
        self.assertEqual(sum(row["phase"] == "warmup" for row in results), 14)
        for row in summary["cells"]:
            self.assertEqual(row["metrics"]["success_count"], row["concurrency"] * 3)

    def test_each_concurrency_level_really_runs_simultaneously(self):
        lock = threading.Lock()
        counts = {False: 0, True: 0}
        barriers = {
            (mode, level): threading.Barrier(level)
            for mode in (False, True)
            for level in (1, 5, 10)
        }

        def behavior(handler, body):
            if len(body["messages"][0]["content"]) != 256:
                mode = enabled(body)
                with lock:
                    counts[mode] += 1
                    n = counts[mode]
                level = 1 if n <= 1 else 5 if n <= 6 else 10
                barriers[(mode, level)].wait(timeout=3)
            completion(handler, body)

        def configure(config):
            config["input_characters"] = [128]

        summary, _, _, _, peaks, _ = self.run_local(behavior, configure)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(peaks[(False, 128)], 10)
        self.assertEqual(peaks[(True, 128)], 10)

    def test_preflight_mode_rejection_does_not_retry_or_block_other_mode(self):
        def behavior(handler, body):
            if not enabled(body):
                handler.send_response(400)
                handler.end_headers()
                handler.wfile.write(SECRET.encode())
            else:
                completion(handler, body)

        summary, _, _, requests, _, _ = self.run_local(behavior)
        self.assertEqual(sum(not enabled(body) for body, _ in requests), 1)
        self.assertFalse(summary["modes"]["off"]["parameters_accepted"])
        self.assertTrue(
            all(row["status"] == "skipped" for row in summary["cells"] if row["mode"] == "off")
        )
        self.assertTrue(
            all(row["status"] == "completed" for row in summary["cells"] if row["mode"] == "on")
        )

    def test_mode_contradiction_detected_in_performance_phase(self):
        def behavior(handler, body):
            completion(
                handler, body, thinking=len(body["messages"][0]["content"]) != 256 or enabled(body)
            )

        summary, _, _, requests, _, _ = self.run_local(behavior)
        self.assertEqual(summary["modes"]["off"]["observation"], "contradicted")
        self.assertEqual(sum(not enabled(body) for body, _ in requests), 2)
        self.assertEqual(summary["status"], "partial")

    def test_unobserved_thinking_is_not_claimed_verified(self):
        def behavior(handler, body):
            completion(handler, body, usage=False, thinking=False)

        summary, _, report, _, _, _ = self.run_local(behavior)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["modes"]["on"]["observation"], "unconfirmed")
        self.assertIn("实际模式未确认", report)

    def test_all_failed_lowest_concurrency_skips_larger_inputs_for_one_mode(self):
        def behavior(handler, body):
            if not enabled(body) and len(body["messages"][0]["content"]) == 128:
                handler.send_response(500)
                handler.end_headers()
            else:
                completion(handler, body)

        summary, _, _, requests, _, _ = self.run_local(behavior)
        off = [row for row in summary["cells"] if row["mode"] == "off"]
        self.assertEqual(off[0]["metrics"]["success_count"], 0)
        self.assertTrue(all(row["status"] == "skipped" for row in off[1:]))
        self.assertFalse(
            any(
                not enabled(body) and len(body["messages"][0]["content"]) == 512
                for body, _ in requests
            )
        )
        self.assertTrue(
            all(row["status"] == "completed" for row in summary["cells"] if row["mode"] == "on")
        )

    def test_failed_high_concurrency_skips_only_that_input(self):
        count = 0
        lock = threading.Lock()

        def behavior(handler, body):
            nonlocal count
            reject = False
            if not enabled(body) and len(body["messages"][0]["content"]) == 128:
                with lock:
                    count += 1
                    reject = count > 1
            if reject:
                handler.send_response(500)
                handler.end_headers()
            else:
                completion(handler, body)

        summary, _, _, _, _, _ = self.run_local(behavior)
        rows = {row["id"]: row for row in summary["cells"]}
        self.assertEqual(rows["off-128-c5"]["metrics"]["success_count"], 0)
        self.assertEqual(rows["off-128-c10"]["status"], "skipped")
        self.assertEqual(rows["off-512-c10"]["status"], "completed")

    def test_partial_failures_continue_to_higher_concurrency(self):
        count = 0
        lock = threading.Lock()

        def behavior(handler, body):
            nonlocal count
            reject = False
            if not enabled(body) and len(body["messages"][0]["content"]) == 128:
                with lock:
                    count += 1
                    reject = count == 2
            if reject:
                handler.send_response(429)
                handler.end_headers()
            else:
                completion(handler, body)

        summary, _, _, _, _, _ = self.run_local(behavior)
        rows = {row["id"]: row for row in summary["cells"]}
        self.assertEqual(rows["off-128-c5"]["metrics"]["success_rate"], 0.8)
        self.assertEqual(rows["off-128-c10"]["status"], "completed")


class WarmupTests(unittest.TestCase):
    run_local = RunnerTests.run_local

    def run_warm(self, behavior=completion, configure=None):
        def settings(config):
            config["warmup"] = {"enabled": True, "requests_per_length": 1}
            if configure:
                configure(config)

        return self.run_local(behavior, settings)

    def test_configuration_validation_and_explicit_disable(self):
        self.assertFalse(config_for()["warmup"]["enabled"])
        self.assertEqual(bench.make_warmups(config_for()), [])
        for value in (
            None,
            True,
            {},
            {"enabled": 1},
            {"enabled": "true"},
            {"enabled": True, "requests_per_length": 0},
            {"enabled": True, "requests_per_length": True},
            {"enabled": False, "requests_per_length": 1001},
            {"enabled": True, "requests_per_length": 1.5},
            {"enabled": True, "requests_per_length": "2"},
            {"enabled": True, "unknown": SECRET},
        ):
            with self.subTest(value=value), self.assertRaises(bench.BenchmarkError) as caught:
                bench.validate_config(dict(config_for(), warmup=value))
            self.assertNotIn(SECRET, str(caught.exception))
        config = bench.validate_config(dict(config_for(), warmup={"enabled": True}))
        self.assertEqual(config["warmup"]["requests_per_length"], 1)
        summary, records, report, requests, _, _ = self.run_warm(
            configure=lambda config: config["warmup"].update(enabled=False)
        )
        self.assertEqual(len(requests), 66)
        self.assertEqual(summary["warmup"]["planned_requests"], 0)
        self.assertIn("本次未启用独立预热", report)
        self.assertFalse(any(r.get("request", {}).get("phase") == "warmup" for r in records))

    def test_default_dry_run_has_14_warmups_and_689_max_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "model.jsonc"
            path.write_text(bench.config_template())
            output = io.StringIO()
            with (
                contextlib.redirect_stdout(output),
                mock.patch.object(
                    bench.http.client.HTTPConnection,
                    "connect",
                    side_effect=AssertionError("network"),
                ),
            ):
                self.assertEqual(bench.main(["--config", str(path), "--dry-run"]), 0)
            plan = json.loads(output.getvalue())
            self.assertEqual(plan["performance_requests"], 672)
            self.assertEqual(plan["warmup_requests"], 14)
            self.assertEqual(plan["total_requests_max"], 689)
            self.assertEqual(list(directory.iterdir()), [path])

    def test_serial_order_once_per_length_and_inherited_request_parameters(self):
        def configure(config):
            config.update(concurrency=[2, 4, 6], output_tokens={"off": 17, "on": 93})
            config["warmup"]["requests_per_length"] = 2

        summary, records, report, bodies, _, snapshot = self.run_warm(configure=configure)
        self.assertEqual(len(bodies), 58)  # 48 performance + 8 warmup + 2 preflight
        expected = []
        for mode in bench.MODES:
            expected += [("preflight", mode, 256)]
            for size in [128, 512]:
                expected += [("warmup", mode, size)] * 2
                expected += [("performance", mode, size)] * 12
        actual = []
        active = set()
        last_phase = None
        requests = []
        for record in records:
            if record["record_type"] not in {"scheduled", "result"}:
                continue
            item = record["request"]
            if record["record_type"] == "scheduled":
                if item["phase"] == "warmup" or last_phase == "warmup":
                    self.assertFalse(active)
                active.add(item["request_id"])
                last_phase = item["phase"]
                actual.append((item["phase"], item["mode"], item["input_characters"]))
                requests.append(item)
            else:
                active.remove(item["request_id"])
        self.assertEqual(actual, expected)
        self.assertEqual(len({r["input_sha256"] for r in requests}), len(requests))
        self.assertEqual(len(snapshot["warmups"]), 4)
        self.assertTrue(all(g["status"] == "completed" for g in summary["warmup"]["groups"]))
        for body, _ in bodies:
            if is_warmup(body):
                self.assertEqual(body["max_tokens"], 93 if enabled(body) else 17)
        self.assertIn("独立预热（不计入性能统计）", report)

    def test_fixed_timings_and_tokens_do_not_contaminate_performance_or_rebuild(self):
        def fake_request(config, spec, body, credential, controller, gate, origin):
            duration, tokens = (200, 9999) if spec["phase"] == "warmup" else (2, 5)
            item = MetricsTests().sample(
                spec["request_id"], spec["batch_id"], 10, 10 + duration, tokens=tokens
            )
            item.update(spec, http_status=200, reasoning_observed=spec["mode"] == "on")
            return item

        with mock.patch.object(bench, "perform_request", side_effect=fake_request):
            summary, _, _, _, _, _ = self.run_warm(
                configure=lambda config: config.update(concurrency=[1])
            )
        self.assertEqual(summary["warmup"]["metrics"]["duration_seconds"], 800)
        self.assertEqual(summary["warmup"]["metrics"]["tokens"]["completion_tokens"]["p50"], 9999)
        for row in summary["cells"]:
            m = row["metrics"]
            self.assertEqual(m["attempted_count"], 1)
            self.assertEqual(m["duration_seconds"], 2)
            self.assertEqual(m["aggregate_output_tps"], 2.5)
            self.assertEqual(m["latency_ms"]["e2e"]["p95"], 2000)
            self.assertEqual(m["output_tps"]["p50"], 2)
            self.assertEqual(m["tpot_ms"]["p50"], 500)

    def test_failed_warmups_are_not_retried_and_performance_continues(self):
        def behavior(handler, body):
            if is_warmup(body):
                handler.send_response(500)
                handler.end_headers()
                handler.wfile.write((SECRET + ANSWER).encode())
            else:
                completion(handler, body)

        summary, _, report, requests, _, _ = self.run_warm(behavior)
        self.assertEqual(len(requests), 70)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["warmup"]["metrics"]["failed_count"], 4)
        self.assertTrue(all(g["status"] == "incomplete" for g in summary["warmup"]["groups"]))
        self.assertTrue(all(row["metrics"]["success_rate"] == 1 for row in summary["cells"]))
        self.assertIn("预热未完成", report)

    def test_warmup_timeout_uses_configured_deadline_then_continues(self):
        def behavior(handler, body):
            if is_warmup(body):
                time.sleep(0.15)
            else:
                completion(handler, body)

        def configure(config):
            config.update(thinking_modes=["off"], input_characters=[128], concurrency=[1])
            config["timeouts"]["read_seconds"] = 0.05

        summary, _, _, requests, _, _ = self.run_warm(behavior, configure)
        self.assertEqual(len(requests), 3)
        self.assertEqual(summary["warmup"]["metrics"]["status_counts"], {"timeout": 1})
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["cells"][0]["metrics"]["success_count"], 1)

    def test_warmup_mode_contradiction_stops_remaining_warmups_and_that_mode(self):
        def behavior(handler, body):
            completion(handler, body, thinking=True if is_warmup(body) else None)

        summary, _, _, requests, _, _ = self.run_warm(
            behavior, lambda config: config["warmup"].update(requests_per_length=2)
        )
        self.assertEqual(sum(not enabled(body) for body, _ in requests), 2)
        self.assertEqual(summary["modes"]["off"]["stop_reason"], "mode_contradicted")
        off = [g for g in summary["warmup"]["groups"] if g["mode"] == "off"]
        self.assertEqual([g["status"] for g in off], ["incomplete", "skipped"])
        self.assertTrue(
            all(row["status"] == "skipped" for row in summary["cells"] if row["mode"] == "off")
        )
        self.assertTrue(
            all(row["status"] == "completed" for row in summary["cells"] if row["mode"] == "on")
        )

    def test_only_reasoning_warmup_is_protocol_success_without_ttfo(self):
        def behavior(handler, body):
            if is_warmup(body):
                start_stream(handler)
                event(
                    handler,
                    {
                        "choices": [
                            {"delta": {"reasoning_content": THOUGHT}, "finish_reason": "length"}
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 4096},
                    },
                )
                event(handler, "[DONE]")
            else:
                completion(handler, body)

        summary, _, _, _, _, _ = self.run_warm(
            behavior, lambda config: config.update(thinking_modes=["on"], concurrency=[1])
        )
        m = summary["warmup"]["metrics"]
        self.assertEqual(m["success_count"], 2)
        self.assertEqual(m["final_answer_count"], 0)
        self.assertIsNone(m["latency_ms"]["ttfo"]["p50"])
        self.assertEqual(summary["status"], "completed")

    def test_interrupt_during_later_warmup_preserves_measurements_and_offline_recovery(self):
        for termination in (signal.SIGINT, signal.SIGKILL):
            active, release = threading.Event(), threading.Event()

            def behavior(handler, body, active=active, release=release):
                if is_warmup(body) and len(body["messages"][0]["content"]) == 512:
                    start_stream(handler)
                    event(handler, {"choices": [{"delta": {"content": ANSWER}}]})
                    active.set()
                    release.wait(10)
                else:
                    completion(handler, body)

            with (
                self.subTest(signal=termination),
                tempfile.TemporaryDirectory() as temporary,
                LocalServer(behavior) as server,
            ):
                directory = Path(temporary)
                config = config_for(server.url)
                config.update(concurrency=[1], warmup={"enabled": True, "requests_per_length": 1})
                config["model"]["api_key"] = SECRET
                config["timeouts"] = {"connect_seconds": 2, "read_seconds": 20, "total_seconds": 30}
                script, path = directory / "llm_benchmark.py", directory / bench.DEFAULT_CONFIG_NAME
                shutil.copyfile(SCRIPT, script)
                path.write_text(bench.config_template(config))
                process = subprocess.Popen(
                    [sys.executable, "-I", "-S", str(script)],
                    cwd=directory,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                try:
                    self.assertTrue(active.wait(5))
                    process.send_signal(termination)
                    stdout, stderr = process.communicate(timeout=4)
                finally:
                    release.set()
                    if process.poll() is None:
                        process.kill()
                        process.communicate(timeout=3)
                self.assertEqual(len(server.requests), 4)
                self.assertNotIn(SECRET, stdout + stderr)
                output = next(directory.glob("llm_benchmark_evidence_*"))
                before = {
                    name: (output / name).read_bytes()
                    for name in ("snapshot.json", "requests.jsonl")
                }
                path.unlink()
                rebuilt = subprocess.run(
                    [sys.executable, "-I", "-S", str(script), "report", "--input", str(output)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.assertEqual(rebuilt.returncode, 0, rebuilt.stderr)
                summary = bench.read_json(output / "summary.json")
                self.assertEqual(summary["cells"][0]["metrics"]["success_count"], 1)
                self.assertEqual(summary["cells"][1]["metrics"]["attempted_count"], 0)
                self.assertEqual(summary["warmup"]["groups"][1]["status"], "incomplete")
                if termination == signal.SIGINT:
                    self.assertEqual(process.returncode, 130, stderr)
                    self.assertEqual(summary["status"], "cancelled")
                    self.assertEqual(summary["warmup"]["metrics"]["status_counts"]["cancelled"], 1)
                else:
                    self.assertEqual(summary["status"], "interrupted")
                    self.assertEqual(summary["warmup"]["metrics"]["unresolved_count"], 1)
                    self.assertIsNone(summary["warmup"]["metrics"]["duration_seconds"])
                persisted = "".join(p.read_text() for p in output.iterdir())
                for secret in (SECRET, ANSWER, THOUGHT):
                    self.assertNotIn(secret, persisted)
                for name, data in before.items():
                    self.assertEqual((output / name).read_bytes(), data)
                self.assertEqual(summary, bench.generate_report(output))

    def test_preflight_and_performance_skip_rules_do_not_schedule_extra_warmups(self):
        for reject_preflight in (True, False):

            def behavior(handler, body, reject_preflight=reject_preflight):
                size = len(body["messages"][0]["content"])
                reject = size == 256 if reject_preflight else size == 128 and not is_warmup(body)
                if not enabled(body) and reject:
                    handler.send_response(400)
                    handler.end_headers()
                else:
                    completion(handler, body)

            with self.subTest(preflight=reject_preflight):
                summary, _, _, requests, _, _ = self.run_warm(behavior)
                off = [g for g in summary["warmup"]["groups"] if g["mode"] == "off"]
                self.assertEqual(
                    [g["status"] for g in off],
                    ["skipped", "skipped"] if reject_preflight else ["completed", "skipped"],
                )
                self.assertEqual(
                    sum(is_warmup(body) and not enabled(body) for body, _ in requests),
                    0 if reject_preflight else 1,
                )


class SelfReviewTests(unittest.TestCase):
    def test_explicit_adapter_applies_to_aliases_and_every_request_phase(self):
        for name, adapter, field in (
            ("production-llm", "qwen", "chat_template_kwargs"),
            ("内部/服务-A", "deepseek", "thinking"),
            ("DeepSeek-V3", "qwen", "chat_template_kwargs"),
            ("Qwen3-8B", "deepseek", "thinking"),
        ):

            def configure(config, name=name, adapter=adapter):
                config["model"].update(name=name, thinking_adapter=adapter)
                config["warmup"] = {"enabled": True, "requests_per_length": 1}

            with self.subTest(name=name), self.completed_run(configure=configure) as run:
                summary, directory, server = run
                self.assertEqual(summary["status"], "completed")
                self.assertEqual(summary["self_review"]["status"], "success")
                self.assertEqual(len(server.requests), 7)
                self.assertEqual(
                    [enabled(body) for body, _ in server.requests],
                    [False, False, False, True, True, True, False],
                )
                for body, _ in server.requests:
                    self.assertEqual(body["model"], name)
                    self.assertIn(field, body)
                    other = "thinking" if field != "thinking" else "chat_template_kwargs"
                    self.assertNotIn(other, body)
                    self.assertNotIn("thinking_adapter", body)
                    self.assertNotIn("extra_body", body)
                snapshot, records, _ = bench.load_evidence(directory)
                metadata = {"requested": adapter, "resolved": adapter, "source": "explicit"}
                self.assertEqual(snapshot["thinking_adapter"], metadata)
                self.assertEqual(summary["thinking_adapter"], metadata)
                scheduled = [r["request"] for r in records if r["record_type"] == "scheduled"]
                self.assertEqual(
                    {r["phase"] for r in scheduled}, {"preflight", "warmup", "performance"}
                )
                for request in scheduled:
                    self.assertEqual(
                        request["mode_parameters"], snapshot["mode_parameters"][request["mode"]]
                    )
                self.assertEqual(
                    summary["self_review"]["request"]["mode_parameters"],
                    snapshot["mode_parameters"]["off"],
                )
                report = (directory / "report.md").read_text()
                self.assertIn(f"{adapter}（配置显式指定；配置值 {adapter}）", report)
                self.assertIn(name, report)
                self.assertEqual(summary, bench.generate_report(directory))
                for filename in (
                    "snapshot.json",
                    "requests.jsonl",
                    "summary.json",
                    "report.md",
                    "self_review.json",
                ):
                    saved = (directory / filename).read_text()
                    for secret in (SECRET, ANSWER, THOUGHT):
                        self.assertNotIn(secret, saved)

    def test_rebuild_uses_frozen_adapter_and_preserves_legacy_review(self):
        for selection in ("auto", "qwen", "legacy"):

            def configure(config, selection=selection):
                if selection == "qwen":
                    config["model"].update(name="production-llm", thinking_adapter="qwen")

            with self.subTest(selection=selection), self.completed_run(configure=configure) as run:
                summary, directory, server = run
                if selection == "legacy":
                    snapshot = bench.read_json(directory / "snapshot.json")
                    snapshot.pop("thinking_adapter")
                    snapshot["config"]["model"].pop("thinking_adapter")
                    bench.atomic_json(directory / "snapshot.json", snapshot)
                originals = {
                    name: (directory / name).read_bytes()
                    for name in ("snapshot.json", "requests.jsonl", "self_review.json")
                }
                with (
                    mock.patch.object(bench, "model_family", side_effect=AssertionError),
                    mock.patch.object(
                        bench, "resolve_thinking_adapter", side_effect=AssertionError
                    ),
                    mock.patch.object(bench, "read_config", side_effect=AssertionError),
                    mock.patch.object(
                        bench.socket, "create_connection", side_effect=AssertionError
                    ),
                ):
                    rebuilt = bench.generate_report(directory)
                    first_report = (directory / "report.md").read_bytes()
                    self.assertEqual(rebuilt, bench.generate_report(directory))
                    self.assertEqual(first_report, (directory / "report.md").read_bytes())
                self.assertEqual(rebuilt["cells"], summary["cells"])
                self.assertEqual(rebuilt["modes"], summary["modes"])
                self.assertEqual(rebuilt["self_review"], summary["self_review"])
                if selection == "legacy":
                    self.assertIsNone(rebuilt["thinking_adapter"])
                    self.assertIn("历史快照未记录方案及选择来源", first_report.decode())
                else:
                    self.assertEqual(rebuilt, summary)
                for name, original in originals.items():
                    self.assertEqual((directory / name).read_bytes(), original)
                self.assertEqual(len(server.requests), 5)

    def test_inconsistent_adapter_snapshot_is_rejected(self):
        with self.completed_run() as (_, directory, _):
            path = directory / "snapshot.json"
            original = bench.read_json(path)
            for metadata in (
                {"requested": "qwen", "resolved": "qwen", "source": "explicit"},
                {"requested": "auto", "resolved": "deepseek", "source": "model_name"},
                {"requested": "auto", "resolved": "qwen", "source": SECRET},
                {"requested": "auto", "resolved": SECRET, "source": "model_name"},
                None,
            ):
                bench.atomic_json(path, dict(original, thinking_adapter=metadata))
                with (
                    self.subTest(metadata=metadata),
                    self.assertRaises(bench.BenchmarkError) as caught,
                ):
                    bench.generate_report(directory)
                self.assertNotIn(SECRET, str(caught.exception))

    def test_warmup_aggregates_remain_separate_in_self_review(self):
        def configure(config):
            config["warmup"] = {"enabled": True, "requests_per_length": 1}

        with self.completed_run(configure=configure) as (summary, directory, server):
            self.assertEqual(len(server.requests), 7)
            review = server.requests[-1][0]
            payload = json.loads(review["messages"][1]["content"])
            self.assertEqual(payload["warmup"]["metrics"]["success_count"], 2)
            self.assertEqual(sum(row["metrics"]["success_count"] for row in payload["cells"]), 2)
            self.assertNotIn(SECRET, json.dumps(payload))
            self.assertNotIn(server.url, json.dumps(payload))
            self.assertEqual(summary["self_review"]["status"], "success")
            self.assertEqual(summary, bench.generate_report(directory))

    def test_family_variants_use_same_switch_for_measurements_and_self_review(self):
        for name, field in (
            ("deepseek-ai/DeepSeek-V3.2-Exp", "thinking"),
            ("org/QWEN3-235B-A22B-Instruct-FP8", "chat_template_kwargs"),
        ):
            with (
                self.subTest(name=name),
                self.completed_run(
                    configure=lambda c, model_name=name: c["model"].update(name=model_name)
                ) as (
                    summary,
                    directory,
                    server,
                ),
            ):
                self.assertEqual(summary["status"], "completed")
                self.assertEqual(summary["self_review"]["status"], "success")
                self.assertEqual(len(server.requests), 5)
                self.assertEqual(
                    [enabled(body) for body, _ in server.requests],
                    [False, False, True, True, False],
                )
                for body, _ in server.requests:
                    self.assertEqual(body["model"], name)
                    self.assertIn(field, body)
                    self.assertNotIn("extra_body", body)
                    other = "thinking" if field != "thinking" else "chat_template_kwargs"
                    self.assertNotIn(other, body)
                self.assertEqual(summary, bench.generate_report(directory))

    def test_run_and_self_review_without_license_at_future_date(self):
        with (
            mock.patch.object(bench.time, "time", return_value=4102444800),
            mock.patch.dict(os.environ, {"PATH": ""}),
            self.completed_run() as (summary, directory, server),
        ):
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["self_review"]["status"], "success")
            self.assertEqual(len(server.requests), 5)
            self.assertEqual(summary, bench.generate_report(directory))
            self.assertNotIn("license_key", (directory / "snapshot.json").read_text())

    def test_rating_order_and_exact_first_line_contract(self):
        ratings = ("夯爆了", "顶级", "人上人", "NPC", "拉完了")
        self.assertEqual(bench.REVIEW_RATINGS, ratings)
        self.assertIn("五档由好到差依次为：" + "、".join(ratings), bench.REVIEW_INSTRUCTIONS)
        self.assertNotIn("不要给自己打分", bench.REVIEW_INSTRUCTIONS)
        for rating in ratings:
            with self.subTest(rating=rating):
                text = "综合档位：" + rating + "\n1. 依据本次已测数据。"
                self.assertEqual(bench.self_review_rating(text), rating)
                report = "\n".join(
                    bench.render_self_review({"status": "success", "rating": rating, "text": text})
                )
                self.assertIn("**综合档位：" + rating + "**", report)
                self.assertEqual(report.count("综合档位："), 1)
                self.assertNotIn("档位由好到差", report)
                self.assertNotIn(" → ".join(ratings), report)
                self.assertIn("```text\n1. 依据本次已测数据。\n```", report)
                rating_only = "\n".join(
                    bench.render_self_review(
                        {"status": "success", "rating": rating, "text": "综合档位：" + rating}
                    )
                )
                self.assertEqual(rating_only.count("综合档位："), 1)
                self.assertNotIn("```", rating_only)
        for text in (
            None,
            "",
            " \n ",
            "综合档位：优秀",
            "综合档位：npc",
            "综合档位：顶级、人上人",
            "**综合档位：顶级**",
            "综合档位：夯爆了（仅供参考）",
            "1. 相比顶级，本次表现为 NPC。",
            "评测如下：\n综合档位：顶级",
        ):
            with self.subTest(text=text):
                self.assertIsNone(bench.self_review_rating(text))

    @contextlib.contextmanager
    def completed_run(
        self,
        review_behavior=review_response,
        configure=None,
        controller=None,
        performance=completion,
    ):
        def behavior(handler, body):
            if is_review(body):
                review_behavior(handler)
            else:
                performance(handler, body)

        with tempfile.TemporaryDirectory() as temporary, LocalServer(behavior) as server:
            config = config_for(server.url)
            config.update(input_characters=[128], concurrency=[1], self_review=True)
            if configure:
                configure(config)
            directory = Path(temporary) / "run"
            with contextlib.redirect_stdout(io.StringIO()):
                summary = bench.run_benchmark(config, directory, SECRET, controller=controller)
            yield summary, directory, server

    def test_explicit_boolean_setting_and_new_templates(self):
        self.assertFalse(config_for()["self_review"])
        for value in (None, 1, "true", {}):
            with self.assertRaises(bench.BenchmarkError):
                bench.validate_config(dict(config_for(), self_review=value))
        for family in ("deepseek", "qwen"):
            self.assertTrue(
                bench.read_config(ROOT / f"llm_benchmark.{family}.example.jsonc")[
                    "self_review"
                ]
            )

    def test_success_uses_only_aggregates_and_does_not_change_measurements(self):
        with self.completed_run() as (summary, directory, server):
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(len(server.requests), 5)
            body = server.requests[-1][0]
            self.assertTrue(is_review(body))
            self.assertFalse(enabled(body))
            self.assertEqual(body["max_tokens"], 2048)
            payload = json.loads(body["messages"][1]["content"])
            self.assertEqual(len(payload["cells"]), 2)
            self.assertEqual(payload["conditions"]["output_tokens"], {"off": 512, "on": 4096})
            prompt = json.dumps(body["messages"], ensure_ascii=False)
            for excluded in (
                SECRET,
                ANSWER,
                THOUGHT,
                server.url,
                "test-seed",
                "api_key",
                "license_key",
                "run_id",
            ):
                self.assertNotIn(excluded, prompt)
            snapshot, records, warnings = bench.load_evidence(directory)
            base = bench.summarize(snapshot, records, warnings)
            self.assertEqual(summary["cells"], base["cells"])
            self.assertEqual(summary["modes"], base["modes"])
            self.assertEqual(records[-1]["record_type"], "run_finished")
            self.assertEqual(sum(r["record_type"] == "result" for r in records), 4)
            self.assertNotIn('"self_review"', (directory / "requests.jsonl").read_text())
            review = summary["self_review"]
            self.assertEqual(review["status"], "success")
            self.assertEqual(review["text"], REVIEW_TEXT)
            self.assertEqual(review["rating"], "人上人")
            self.assertEqual(review["prompt_version"], "performance-self-review/v2")
            self.assertIn("综合档位：<档位>", body["messages"][0]["content"])
            self.assertEqual(bench.read_json(directory / "self_review.json")["rating"], "人上人")
            self.assertEqual(review["source_sha256"], bench.self_review_input(base)[1])
            before = (directory / "report.md").read_bytes()
            self.assertEqual(summary, bench.generate_report(directory))
            self.assertEqual(before, (directory / "report.md").read_bytes())
            self.assertEqual(len(server.requests), 5)
            report_body, separator, footer = (
                (directory / "report.md").read_text().rpartition("\n\n---\n\n")
            )
            self.assertTrue(separator)
            self.assertIn(REVIEW_TEXT.partition("\n")[2] + "\n```", report_body)
            self.assertLess(
                report_body.index("## 模型自评"), report_body.index("## 测量口径与附录")
            )
            self.assertIn(bench.SUPPORT_URL, footer)
            self.assertEqual((directory / "report.md").read_text().count("综合档位："), 1)
            with contextlib.redirect_stdout(io.StringIO()):
                bench.write_self_review(summary, directory, SECRET, bench.StopController())
            self.assertEqual(len(server.requests), 5)

    def test_support_footer_is_offline_and_separate_from_evidence_and_model_input(self):
        for with_review in (False, True):
            with (
                self.subTest(self_review=with_review),
                self.completed_run(
                    configure=lambda c, enabled=with_review: c.update(self_review=enabled)
                ) as (summary, directory, server),
            ):
                report = (directory / "report.md").read_bytes()
                footer = report.decode().rpartition("\n\n---\n\n")[2]
                self.assertEqual(report.decode().count(bench.SUPPORT_URL), 1)
                self.assertIn("请作者喝一杯瑞幸咖啡", footer)
                self.assertIn("支持全凭自愿，不影响任何功能的使用", footer)
                self.assertNotIn("<img", footer)
                for path in directory.glob("*.json*"):
                    self.assertNotIn(bench.SUPPORT_URL, path.read_text())
                for body, _ in server.requests:
                    self.assertNotIn(bench.SUPPORT_URL, json.dumps(body))
                before = {path.name: path.read_bytes() for path in directory.glob("*.json*")}
                with mock.patch.object(
                    bench.socket,
                    "create_connection",
                    side_effect=AssertionError("unexpected network"),
                ):
                    self.assertEqual(summary, bench.generate_report(directory))
                    self.assertEqual(summary, bench.generate_report(directory))
                self.assertEqual(report, (directory / "report.md").read_bytes())
                self.assertEqual(
                    before, {path.name: path.read_bytes() for path in directory.glob("*.json*")}
                )
                self.assertEqual(len(server.requests), 5 if with_review else 4)

    def test_unrecognized_rating_keeps_text_and_does_not_retry(self):
        text = "综合档位：优秀\n1. 已测并发不能代表绝对容量。"
        with self.completed_run(lambda h: review_response(h, text)) as (
            summary,
            directory,
            server,
        ):
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["self_review"]["status"], "success")
            self.assertIsNone(summary["self_review"]["rating"])
            self.assertEqual(summary["self_review"]["text"], text)
            report = (directory / "report.md").read_text()
            self.assertIn("档位未识别", report)
            self.assertNotIn("**综合档位：", report)
            self.assertEqual(summary, bench.generate_report(directory))
            self.assertEqual(len(server.requests), 5)

    def test_inconsistent_rating_evidence_never_changes_offline_metrics(self):
        with self.completed_run() as (summary, directory, server):
            path = directory / "self_review.json"
            original = bench.read_json(path)
            for rating in ("夯爆了", "优秀", None, [], {"label": "人上人"}):
                with self.subTest(rating=rating):
                    bench.atomic_json(path, dict(original, rating=rating))
                    rebuilt = bench.generate_report(directory)
                    self.assertEqual(rebuilt["self_review"]["status"], "invalid")
                    self.assertEqual(rebuilt["cells"], summary["cells"])
                    self.assertNotIn(REVIEW_TEXT, (directory / "report.md").read_text())
            self.assertEqual(len(server.requests), 5)

    def test_disabled_and_missing_success_skip_without_extra_requests(self):
        with self.completed_run(configure=lambda c: c.update(self_review=False)) as (
            summary,
            directory,
            server,
        ):
            self.assertEqual(summary["self_review"]["status"], "disabled")
            self.assertEqual(len(server.requests), 4)
            self.assertFalse((directory / "self_review.json").exists())
            self.assertNotIn("## 模型自评", (directory / "report.md").read_text())

        def reject(handler, body):
            handler.send_response(401)
            handler.end_headers()

        with self.completed_run(performance=reject) as (summary, _, server):
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["self_review"]["reason"], "no_successful_measurements")
            self.assertFalse(any(is_review(body) for body, _ in server.requests))

    def test_known_off_mode_contradiction_skips_review(self):
        def behavior(handler, body):
            completion(handler, body, thinking=True)

        with self.completed_run(performance=behavior) as (summary, _, server):
            self.assertEqual(summary["self_review"]["reason"], "off_mode_unavailable")
            self.assertFalse(any(is_review(body) for body, _ in server.requests))

    def test_http_invalid_sse_and_reasoning_failures_preserve_report(self):
        def rejected(handler):
            handler.send_response(400)
            handler.end_headers()
            handler.wfile.write(SECRET.encode())

        def invalid(handler):
            start_stream(handler)
            event(handler, "not JSON " + SECRET)

        def thinking(handler):
            start_stream(handler)
            for part in ("<thi", "nk>" + THOUGHT, "</thi", "nk>" + REVIEW_TEXT):
                event(handler, {"choices": [{"delta": {"content": part}}]})
            event(handler, "[DONE]")

        for behavior, reason in (
            (rejected, "http_error"),
            (invalid, "protocol_error"),
            (thinking, "mode_contradicted"),
        ):
            with (
                self.subTest(reason=reason),
                self.completed_run(review_behavior=behavior) as (summary, directory, server),
            ):
                self.assertEqual(summary["status"], "completed")
                self.assertEqual(summary["self_review"]["status"], "failed")
                self.assertEqual(summary["self_review"]["reason"], reason)
                self.assertIsNone(summary["self_review"]["text"])
                self.assertEqual(len(server.requests), 5)
                persisted = "".join(p.read_text() for p in directory.iterdir())
                for value in (SECRET, THOUGHT, ANSWER):
                    self.assertNotIn(value, persisted)
                self.assertIn("生成失败", (directory / "report.md").read_text())

    def test_timeout_and_cancel_during_review_do_not_erase_completed_run(self):
        for cancel in (False, True):
            controller = bench.StopController()

            def slow(handler, should_cancel=cancel, stop=controller):
                start_stream(handler)
                if should_cancel:
                    stop.cancel()
                time.sleep(0.3)

            def configure(config):
                config["timeouts"]["read_seconds"] = 0.15

            with (
                self.subTest(cancel=cancel),
                self.completed_run(slow, configure, controller) as (summary, directory, _),
            ):
                self.assertEqual(summary["status"], "completed")
                self.assertEqual(
                    summary["self_review"]["reason"], "cancelled" if cancel else "timeout"
                )
                self.assertEqual(summary, bench.generate_report(directory))

    def test_truncated_review_escapes_markup_and_redacts_known_credentials(self):
        hostile = REVIEW_TEXT + "\n```\n# replace report\n<img src=x>\n" + SECRET
        with self.completed_run(lambda h: review_response(h, hostile, "length")) as (
            summary,
            directory,
            _,
        ):
            self.assertEqual(summary["status"], "completed")
            report = (directory / "report.md").read_text()
            self.assertIn("可能不完整", report)
            self.assertIn("````text", report)
            for path in directory.iterdir():
                persisted = path.read_text()
                self.assertNotIn(SECRET, persisted)
            self.assertIn("[已脱敏]", summary["self_review"]["text"])

    @unittest.skipUnless(os.name == "posix", "POSIX interruption signals")
    def test_process_interruption_during_review_rebuilds_without_config_or_network(self):
        for signum in (signal.SIGINT, signal.SIGKILL):
            reviewing, release = threading.Event(), threading.Event()

            def behavior(handler, body, ready=reviewing, done=release):
                if is_review(body):
                    start_stream(handler)
                    ready.set()
                    done.wait(timeout=5)
                else:
                    completion(handler, body)

            with (
                self.subTest(signum=signum),
                tempfile.TemporaryDirectory() as temporary,
                LocalServer(behavior) as server,
            ):
                directory = Path(temporary)
                script = directory / "llm_benchmark.py"
                shutil.copyfile(SCRIPT, script)
                config = config_for(server.url)
                config.update(
                    input_characters=[128],
                    concurrency=[1],
                    self_review=True,
                )
                config["model"]["api_key"] = SECRET
                path = directory / bench.DEFAULT_CONFIG_NAME
                path.write_text(bench.config_template(config))
                process = subprocess.Popen(
                    [sys.executable, "-I", "-S", str(script)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                try:
                    self.assertTrue(reviewing.wait(timeout=5))
                    process.send_signal(signum)
                    stdout, stderr = process.communicate(timeout=5)
                    self.assertEqual(
                        process.returncode, 130 if signum == signal.SIGINT else -signal.SIGKILL
                    )
                    self.assertNotIn(SECRET, stdout + stderr)
                finally:
                    release.set()
                    if process.poll() is None:
                        process.kill()
                        process.communicate(timeout=5)
                path.unlink()
                evidence = next(directory.glob("llm_benchmark_evidence_*"))
                result = subprocess.run(
                    [sys.executable, "-I", "-S", str(script), "report", "--input", str(evidence)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                summary = bench.read_json(evidence / "summary.json")
                self.assertEqual(summary["status"], "completed")
                self.assertEqual(
                    summary["self_review"]["status"],
                    "failed" if signum == signal.SIGINT else "incomplete",
                )
                self.assertEqual(len(server.requests), 5)

    def test_damaged_pending_and_stale_review_never_break_offline_metrics(self):
        with self.completed_run() as (summary, directory, server):
            path = directory / "self_review.json"
            original = bench.read_json(path)
            for update, expected in (
                ({"status": "pending", "text": None, "rating": None}, "incomplete"),
                ({"source_sha256": "0" * 64}, "stale"),
            ):
                bench.atomic_json(path, dict(original, **update))
                rebuilt = bench.generate_report(directory)
                self.assertEqual(rebuilt["self_review"]["status"], expected)
                self.assertEqual(rebuilt["cells"], summary["cells"])
            path.write_text("invalid JSON")
            rebuilt = bench.generate_report(directory)
            self.assertEqual(rebuilt["self_review"]["status"], "invalid")
            self.assertEqual(rebuilt["status"], "completed")
            self.assertEqual(len(server.requests), 5)


class ProcessAndEvidenceTests(unittest.TestCase):
    def test_unknown_family_stops_before_network_and_evidence_creation(self):
        with tempfile.TemporaryDirectory() as temporary, LocalServer() as server:
            root = Path(temporary)
            config = config_for(server.url)
            config["model"].update(name="unknown-model", api_key=SECRET)
            path, output = root / "config.json", root / "run"
            path.write_text(json.dumps(config))
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = bench.main(["run", "--config", str(path), "--output", str(output)])
            self.assertEqual(code, 2)
            self.assertIn("无法识别模型系列", stderr.getvalue())
            self.assertNotIn(SECRET, stderr.getvalue())
            self.assertEqual(server.requests, [])
            self.assertFalse(output.exists())

    def test_obsolete_license_field_has_actionable_redacted_error(self):
        raw = dict(config_for(), license_key=SECRET)
        with self.assertRaises(bench.BenchmarkError) as caught:
            bench.validate_config(raw)
        self.assertIn("license_key 已移除", str(caught.exception))
        self.assertNotIn(SECRET, str(caught.exception))
        self.assertNotIn("license_key", bench.config_template())

    def test_default_json_fallback_and_jsonc_priority_preserve_existing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            script = directory / "llm_benchmark.py"
            shutil.copyfile(SCRIPT, script)
            original = directory / "llm_benchmark.json"
            original.write_text(json.dumps(config_for()))
            before = original.read_bytes()
            args = [sys.executable, "-I", "-S", str(script), "--dry-run"]
            result = subprocess.run(args, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["thinking_modes"], ["off", "on"])
            preferred = directory / bench.DEFAULT_CONFIG_NAME
            preferred.write_text(bench.config_template(dict(config_for(), thinking_modes=["on"])))
            preferred_before = preferred.read_bytes()
            result = subprocess.run(args, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["thinking_modes"], ["on"])
            result = subprocess.run(
                args + ["--config", str(original)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["thinking_modes"], ["off", "on"])
            self.assertEqual(original.read_bytes(), before)
            self.assertEqual(preferred.read_bytes(), preferred_before)

    def test_no_arguments_uses_sibling_config_and_publishes_sibling_report(self):
        with tempfile.TemporaryDirectory() as temporary, LocalServer() as server:
            root = Path(temporary)
            script_dir, work_dir = root / "tool", root / "elsewhere"
            script_dir.mkdir()
            work_dir.mkdir()
            script = script_dir / "llm_benchmark.py"
            shutil.copyfile(SCRIPT, script)
            raw = {
                "schema_version": bench.CONFIG_VERSION,
                "model": {"name": "Qwen3.8-27B", "api_url": server.url, "api_key": SECRET},
            }
            config_path = script_dir / bench.DEFAULT_CONFIG_NAME
            config_path.write_text(json.dumps(raw))
            before_config = config_path.read_bytes()
            result = subprocess.run(
                [sys.executable, "-I", "-S", str(script)],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=40,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(server.requests), 674)
            reports = list(script_dir.glob("llm_benchmark_report_*.md"))
            directories = list(script_dir.glob("llm_benchmark_evidence_*"))
            self.assertEqual(len(reports), 1)
            self.assertEqual(len(directories), 1)
            self.assertEqual(list(work_dir.iterdir()), [])
            self.assertEqual(config_path.read_bytes(), before_config)
            self.assertEqual(reports[0].read_bytes(), (directories[0] / "report.md").read_bytes())
            output_text = reports[0].read_text() + result.stdout + result.stderr
            output_text += "".join(path.read_text() for path in directories[0].iterdir())
            self.assertNotIn(SECRET, output_text)
            self.assertNotIn("api_key", (directories[0] / "snapshot.json").read_text())
            summary = bench.read_json(directories[0] / "summary.json")
            self.assertEqual(summary["planned_performance_requests"], 672)
            self.assertEqual(summary["status"], "completed")
            dry_run = subprocess.run(
                [sys.executable, "-I", "-S", str(script), "--dry-run"],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(dry_run.returncode, 0, dry_run.stderr)
            self.assertNotIn(SECRET, dry_run.stdout + dry_run.stderr)
            self.assertEqual(len(list(script_dir.glob("llm_benchmark_report_*.md"))), 1)
            config_path.unlink()
            original_report = reports[0].read_bytes()
            reports[0].unlink()
            bench.generate_report(directories[0])
            self.assertEqual(reports[0].read_bytes(), original_report)

    def test_first_run_creates_only_sibling_template_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script_dir = root / "tool"
            script_dir.mkdir()
            script = script_dir / "llm_benchmark.py"
            shutil.copyfile(SCRIPT, script)
            args = [sys.executable, "-I", "-S", str(script)]
            result = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 2)
            config = script_dir / bench.DEFAULT_CONFIG_NAME
            before = config.read_bytes()
            self.assertEqual(
                set(bench.read_config(config)["model"]),
                {"name", "thinking_adapter", "api_url", "api_key"},
            )
            self.assertEqual(bench.read_config(config)["model"]["thinking_adapter"], "auto")
            self.assertEqual(bench.read_config(config)["model"]["api_key"], "")
            result = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 2)
            self.assertIn("model.api_key", result.stderr)
            self.assertEqual(config.read_bytes(), before)
            self.assertEqual(
                sorted(path.name for path in script_dir.iterdir()),
                [bench.DEFAULT_CONFIG_NAME, "llm_benchmark.py"],
            )

    def test_report_filename_cannot_overwrite_configuration(self):
        for name in ("../llm_benchmark.json", "llm_benchmark.json", "/tmp/report.md"):
            with self.assertRaises(bench.BenchmarkError):
                bench.validate_report_name(name)

    def test_isolated_script_runs_then_rebuilds_without_key(self):
        for name in (
            "DeepSeek-V4-Flash-0731",
            "deepseek-ai/DeepSeek-V3.2-Exp",
            "Qwen/Qwen3.8-27B",
            "Qwen3.6-35B-A3B",
            "org/QWEN3-8B-AWQ",
        ):
            with (
                self.subTest(name=name),
                tempfile.TemporaryDirectory() as temporary,
                LocalServer() as server,
            ):
                directory = Path(temporary)
                shutil.copyfile(SCRIPT, directory / "benchmark.py")
                config = config_for(server.url, name)
                config.update(input_characters=[128], concurrency=[1], repetitions=1)
                config["model"]["api_key"] = SECRET
                (directory / "model.json").write_text(json.dumps(config))
                args = [sys.executable, "-I", "-S", "benchmark.py"]
                result = subprocess.run(
                    args + ["run", "--config", "model.json", "--output", "run"],
                    cwd=directory,
                    env=dict(
                        os.environ,
                        HTTP_PROXY="http://127.0.0.1:1",
                        HTTPS_PROXY="http://127.0.0.1:1",
                    ),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                before = (directory / "run/report.md").read_bytes()
                outside_reports = list(directory.glob("llm_benchmark_report_*.md"))
                self.assertEqual(len(outside_reports), 1)
                self.assertEqual(outside_reports[0].read_bytes(), before)
                (directory / "model.json").unlink()
                result = subprocess.run(
                    args + ["report", "--input", "run"],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(before, (directory / "run/report.md").read_bytes())
                self.assertNotIn(SECRET, before.decode())
                self.assertEqual(len(server.requests), 4)
                self.assertTrue(all(body["model"] == name for body, _ in server.requests))

    @unittest.skipUnless(hasattr(signal, "SIGKILL"), "POSIX signal acceptance")
    def test_interrupt_and_force_kill_preserve_recoverable_evidence(self):
        for terminate_signal in (signal.SIGINT, signal.SIGKILL):
            active = threading.Event()
            release = threading.Event()

            def behavior(handler, body, active=active, release=release):
                if len(body["messages"][0]["content"]) == 256:
                    completion(handler, body)
                else:
                    start_stream(handler)
                    event(handler, {"choices": [{"delta": {"content": ANSWER}}]})
                    active.set()
                    release.wait(10)

            with (
                self.subTest(signal=terminate_signal),
                tempfile.TemporaryDirectory() as temporary,
                LocalServer(behavior) as server,
            ):
                directory = Path(temporary)
                config = config_for(server.url)
                config["timeouts"] = {"connect_seconds": 2, "read_seconds": 20, "total_seconds": 30}
                config["model"]["api_key"] = SECRET
                script = directory / "benchmark.py"
                shutil.copyfile(SCRIPT, script)
                path = directory / "config.json"
                path.write_text(json.dumps(config))
                output = directory / "run"
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-I",
                        "-S",
                        str(script),
                        "run",
                        "--config",
                        str(path),
                        "--output",
                        str(output),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                try:
                    self.assertTrue(active.wait(5))
                    process.send_signal(terminate_signal)
                    stdout, stderr = process.communicate(timeout=4)
                finally:
                    release.set()
                    if process.poll() is None:
                        process.kill()
                        process.communicate(timeout=3)
                self.assertNotIn(SECRET, stdout + stderr)
                recovered = bench.generate_report(output)
                if terminate_signal == signal.SIGINT:
                    self.assertEqual(process.returncode, 130, stderr)
                    self.assertEqual(recovered["status"], "cancelled")
                    self.assertEqual(
                        recovered["cells"][0]["metrics"]["status_counts"]["cancelled"], 1
                    )
                else:
                    self.assertEqual(recovered["status"], "interrupted")
                    self.assertEqual(recovered["cells"][0]["metrics"]["unresolved_count"], 1)
                    self.assertIsNone(recovered["cells"][0]["metrics"]["success_rate"])
                self.assertTrue((output / "report.md").is_file())
                outside = list(directory.glob("llm_benchmark_report_*.md"))
                self.assertEqual(len(outside), 1)
                self.assertEqual(outside[0].read_bytes(), (output / "report.md").read_bytes())
                outside_html = list(directory.glob("llm_benchmark_report_*.html"))
                self.assertEqual(len(outside_html), 1)
                self.assertEqual(
                    outside_html[0].read_bytes(), (output / "report.html").read_bytes()
                )
                self.assertIn(recovered["status"], (output / "report.html").read_text())

    def test_incomplete_last_record_is_visible_and_middle_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, LocalServer() as server:
            config = config_for(server.url)
            config.update(input_characters=[128], concurrency=[1])
            output = Path(temporary) / "run"
            with contextlib.redirect_stdout(io.StringIO()):
                bench.run_benchmark(config, output, SECRET)
            journal = output / "requests.jsonl"
            original = journal.read_bytes()
            journal.write_bytes(original + b'{"interrupted')
            report = bench.generate_report(output)
            self.assertEqual(report["status"], "interrupted")
            self.assertTrue(report["warnings"])
            self.assertEqual(journal.read_bytes(), original + b'{"interrupted')
            journal.write_bytes(b"invalid\n" + original)
            with self.assertRaises(bench.BenchmarkError):
                bench.generate_report(output)

    def test_no_secret_echo_on_invalid_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            config = config_for()
            config["api_key"] = SECRET
            path.write_text(json.dumps(config))
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    str(SCRIPT),
                    "run",
                    "--config",
                    str(path),
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(result.returncode, 2)
            self.assertNotIn(SECRET, result.stdout + result.stderr)


class ReportFormatTests(unittest.TestCase):
    def test_report_css_matches_accepted_style_baseline(self):
        root = ROOT / "tests/fixtures/bench-report-style"
        base = (root / "base.css").read_text(encoding="utf-8")
        agent = (root / "agent-print.css").read_text(encoding="utf-8")
        for enabled in (False, True):
            with self.subTest(agent=enabled):
                self.assertEqual(
                    bench.report_styles(enabled),
                    base + (agent if enabled else ""),
                    "报告样式偏离已确认基线；有意调整时同步更新 "
                    "docs/design/bench-report-style/README.md 的版本与变更记录及样式快照。",
                )

    @staticmethod
    def style_chart_samples():
        # Synthetic layout-only data: no real model, endpoint, payload or run identifiers.
        def point(value, index):
            return {
                "value": value,
                "flagged": True,
                "source": "style-sample-" + str(index),
                "target": bench.detail_anchor("style-sample-" + str(index)),
                "marker": "A" if index == 3 else "",
                "note": "有效样本 3；截断 1",
            }

        points = [point(value, index) for index, value in enumerate([0, 4, None, 8])]
        return {
            "matrix.html": bench.html_matrix_chart(
                "最终回答等待 · TTFO P50（秒，越低越好）",
                ["1,024 字符 · 并发 1", "2,048 字符 · 并发 1"],
                [("关闭思考", points[:2]), ("开启思考", points[2:])],
            ),
            "line.html": bench.html_line_chart(
                "首 Token 等待 · TTFT P95（秒，越低越好）",
                [("并发 1", list(zip([1024, 2048, 4096, 8192], points)))],
            ),
        }

    def test_svg_matches_accepted_style_baseline(self):
        root = ROOT / "tests/fixtures/bench-report-style"
        for name, document in self.style_chart_samples().items():
            with self.subTest(chart=name):
                self.assertEqual(
                    document,
                    (root / name).read_text(encoding="utf-8"),
                    "图表样式偏离已确认基线；请核对坐标、缺失值、重点标记与链接，"
                    "有意调整时同步更新样式规范和快照。",
                )

    def test_html_only_run_cli_reports_only_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary, LocalServer() as server:
            directory = Path(temporary)
            raw = config_for(server.url)
            raw.update(
                report_formats=["html"],
                thinking_modes=["off"],
                input_characters=[128],
                concurrency=[1],
                repetitions=1,
            )
            raw["model"]["api_key"] = SECRET
            path = directory / "model.jsonc"
            path.write_text(bench.config_template(raw))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(
                    bench.main(["--config", str(path), "--output", str(directory / "run")]), 0
                )
            self.assertNotIn(".md", stdout.getvalue())
            files = list(directory.glob("llm_benchmark_report_*.html"))
            self.assertEqual(len(files), 1)
            self.assertIn(str(files[0]), stdout.getvalue())
            self.assertFalse(list(directory.glob("*.md")))
            self.assertFalse((directory / "run/report.md").exists())

    @contextlib.contextmanager
    def completed(self, formats=None, behavior=completion, **settings):
        with tempfile.TemporaryDirectory() as temporary, LocalServer(behavior) as server:
            directory = Path(temporary) / "run"
            config = config_for(server.url)
            config.update(input_characters=[128], concurrency=[1], repetitions=3)
            config.update(settings)
            if formats is not None:
                config["report_formats"] = formats
            summary = bench.run_benchmark(
                config, directory, SECRET, report_file="llm_benchmark_report_formats.md"
            )
            yield directory, summary, server

    def test_defaults_examples_and_commented_formats(self):
        raw = {"schema_version": bench.CONFIG_VERSION, "model": dict(bench.DEFAULT_MODEL)}
        self.assertEqual(bench.validate_config(raw)["report_formats"], ["md", "html"])
        for source in [bench.config_template()] + [
            (ROOT / f"llm_benchmark.{name}.example.jsonc").read_text()
            for name in ("qwen", "deepseek")
        ]:
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "model.jsonc"
                for disabled, expected in (
                    (None, ["md", "html"]),
                    ("md", ["html"]),
                    ("html", ["md"]),
                ):
                    text = (
                        source
                        if disabled is None
                        else source.replace(f'    "{disabled}",', f'    // "{disabled}",')
                    )
                    path.write_text(text)
                    config = bench.validate_config(bench.read_config(path))
                    self.assertEqual(config["report_formats"], expected)
                    stdout = io.StringIO()
                    with contextlib.redirect_stdout(stdout):
                        self.assertEqual(bench.main(["--config", str(path), "--dry-run"]), 0)
                    self.assertEqual(json.loads(stdout.getvalue())["report_formats"], expected)
                path.write_text(
                    source.replace('    "md",', '    // "md",').replace(
                        '    "html",', '    // "html",'
                    )
                )
                with self.assertRaises(bench.BenchmarkError):
                    bench.validate_config(bench.read_config(path))

    def test_invalid_formats_rejected_without_leaking_input(self):
        for value in ([], None, "md", True, ["pdf"], ["MD"], ["md", "md"], [1], [{}], [SECRET]):
            with self.subTest(value=value), self.assertRaises(bench.BenchmarkError) as caught:
                bench.validate_config(dict(config_for(), report_formats=value))
            self.assertNotIn(SECRET, str(caught.exception))

    def test_selected_formats_and_offline_rebuild(self):
        for formats in (["md"], ["html"], ["md", "html"]):
            with (
                self.subTest(formats=formats),
                self.completed(formats) as (directory, summary, server),
            ):
                self.assertEqual(summary["config"]["report_formats"], formats)
                for extension in bench.REPORT_FORMATS:
                    inner = directory / ("report." + extension)
                    outer = directory.parent / ("llm_benchmark_report_formats." + extension)
                    self.assertEqual(inner.exists(), extension in formats)
                    self.assertEqual(outer.exists(), extension in formats)
                    if extension in formats:
                        self.assertEqual(inner.read_bytes(), outer.read_bytes())
                before = {p.name: p.read_bytes() for p in directory.iterdir()}
                request_count = len(server.requests)
                stdout = io.StringIO()
                with (
                    mock.patch.object(
                        bench.socket, "create_connection", side_effect=AssertionError
                    ),
                    contextlib.redirect_stdout(stdout),
                ):
                    self.assertEqual(bench.main(["report", "--input", str(directory)]), 0)
                self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})
                self.assertEqual(len(server.requests), request_count)
                for extension in bench.REPORT_FORMATS:
                    self.assertEqual(
                        "report." + extension in stdout.getvalue(), extension in formats
                    )
                self.assertNotIn(SECRET, "".join(p.read_text() for p in directory.iterdir()))
                self.assertNotIn(ANSWER, "".join(p.read_text() for p in directory.iterdir()))

    def test_old_snapshot_keeps_markdown_only_and_saved_review(self):
        def behavior(handler, body):
            return review_response(handler) if is_review(body) else completion(handler, body)

        with self.completed(["md"], behavior, self_review=True) as (directory, summary, _):
            path = directory / "snapshot.json"
            snapshot = bench.read_json(path)
            del snapshot["config"]["report_formats"]
            snapshot["script_version"] = "1.8.0"
            bench.atomic_json(path, snapshot)
            old = path.read_bytes()
            rebuilt = bench.generate_report(directory)
            self.assertEqual(rebuilt["config"]["report_formats"], ["md"])
            self.assertEqual(summary["self_review"], rebuilt["self_review"])
            self.assertFalse((directory / "report.html").exists())
            self.assertEqual(path.read_bytes(), old)

    def test_html_content_is_inert_and_review_precedes_appendix(self):
        from html.parser import HTMLParser

        class Inspector(HTMLParser):
            def __init__(self):
                super().__init__()
                self.tags = []
                self.text = []

            def handle_starttag(self, tag, attrs):
                self.tags.append(tag)
                self.assert_safe(attrs)

            def assert_safe(self, attrs):
                for key, _ in attrs:
                    if key.startswith("on") or key == "src":
                        raise AssertionError("active content")

            def handle_data(self, data):
                self.text.append(data)

        text = (
            "综合档位：人上人\n<script>alert(1)</script>\n```\n"
            '<img src="https://bad.invalid/x"> & **原文**'
        )

        def behavior(handler, body):
            return (
                review_response(handler, text=text)
                if is_review(body)
                else completion(handler, body)
            )

        with self.completed(["md", "html"], behavior, self_review=True) as (directory, summary, _):
            document = (directory / "report.html").read_text()
            parser = Inspector()
            parser.feed(document)
            self.assertFalse(
                set(parser.tags) & {"script", "img", "iframe", "input", "select", "details"}
            )
            self.assertIn(text.split("\n", 1)[1], "".join(parser.text))
            self.assertEqual(document.count("综合档位："), 1)
            self.assertLess(document.index("<h2>模型自评"), document.index("<h2>测量口径与附录"))
            markdown = (directory / "report.md").read_text()
            self.assertLess(markdown.index("## 模型自评"), markdown.index("## 测量口径与附录"))
            self.assertIn("A4 portrait", document)
            self.assertIn("table-header-group", document)
            for row in summary["cells"]:
                for key in ("ttft", "ttfo", "e2e"):
                    self.assertIn(bench.fmt(row["metrics"]["latency_ms"][key]["p95"]), document)

    def test_html_single_mode_partial_and_missing_usage(self):
        def behavior(handler, body):
            completion(handler, body, usage=False)

        with self.completed(["html"], behavior, thinking_modes=["off"]) as (directory, summary, _):
            document = (directory / "report.html").read_text()
            self.assertNotIn("<h2>双模式对照", document)
            self.assertIn("N/A", document)
            summary["status"] = "cancelled"
            row = summary["cells"][0]
            row.update(status="partial", reason="cancelled")
            row["metrics"]["unresolved_count"] = 1
            row["metrics"]["error_counts"] = {"<script>unsafe</script>": 1}
            row["metrics"]["aggregate_output_tps"] = None
            document = bench.render_html_report(summary)
            self.assertIn("cancelled", document)
            self.assertIn("partial", document)
            self.assertIn("&lt;script&gt;unsafe&lt;/script&gt;", document)
            self.assertNotIn("<script>", document)
            self.assertIn("未决性能请求", document)

    def test_observations_distinguish_local_recovery_and_insufficient_evidence(self):
        with self.completed(["md"], thinking_modes=["off"]) as (_, summary, _):
            original = summary["cells"][0]
            summary["cells"] = []
            for size, value in ((128, 100), (256, 150), (512, 90)):
                row = copy.deepcopy(original)
                row["input_characters"] = size
                row["metrics"]["latency_ms"]["ttft"]["p95"] = value
                row["metrics"]["latency_ms"]["e2e"]["p95"] = 100
                row["metrics"]["aggregate_output_tps"] = 100
                summary["cells"].append(row)
            observations = "\n".join(bench.performance_observations(summary))
            self.assertIn("50.0%", observations)
            self.assertIn("局部劣化", observations)
            self.assertIn("128 到 256 字符", observations)
            self.assertIn("实际输出 Token", observations)
            summary["cells"][2]["metrics"]["latency_ms"]["ttft"]["p95"] = 140
            self.assertIn(
                "需复测确认是否持续劣化", "\n".join(bench.performance_observations(summary))
            )
            summary["cells"] = summary["cells"][:2]
            self.assertIn("缺少完整的下一档", "\n".join(bench.performance_observations(summary)))
            summary["cells"][0]["metrics"]["latency_ms"]["ttft"]["p95"] = 0
            self.assertIn("未筛出", "\n".join(bench.performance_observations(summary)))
            summary["cells"][0]["metrics"]["latency_ms"]["ttft"]["p95"] = 100
            summary["cells"][1]["status"] = "partial"
            self.assertIn("未筛出", "\n".join(bench.performance_observations(summary)))

    def test_conclusions_precede_details_and_preserve_incomplete_answers(self):
        with self.completed(["md", "html"], thinking_modes=["off"]) as (_, summary, _):
            metrics = summary["cells"][0]["metrics"]
            metrics.update(truncated_count=2, final_answer_count=1, unknown_answer_count=1)
            for document in (bench.render_report(summary), bench.render_html_report(summary)):
                self.assertLess(document.index("关键结论"), document.index("测试条件"))
                self.assertLess(document.index("性能变化分析"), document.index("测试条件"))
                self.assertIn("截断 2 次，未产生最终回答 1 次", document)
                self.assertIn("最终回答状态未知 1 次", document)

    def test_review_sections_preserve_words_and_escape_model_markup(self):
        from html.parser import HTMLParser

        class Text(HTMLParser):
            def __init__(self):
                super().__init__()
                self.values = []

            def handle_data(self, data):
                self.values.append(data)

        body = "1. 首句。后文 <script>alert(1)</script>\n\n2. 第二项\n原文 **强调**。"
        review = {
            "status": "success",
            "rating": "人上人",
            "text": "综合档位：人上人\n" + body,
            "request": {"truncated": True},
        }
        document = "".join(bench.render_self_review_html(review))
        text = Text()
        text.feed(document)
        self.assertIn("".join(body.split()), "".join("".join(text.values).split()))
        self.assertEqual(document.count('class="review-section"'), 2)
        self.assertEqual(document.count("综合档位："), 1)
        self.assertIn("以下内容可能不完整", document)
        self.assertNotIn("<script>", document)
        self.assertIn("&lt;script&gt;", document)
        fallback = bench.self_review_body_html("1. 一。\n3. 三。")
        self.assertNotIn('class="review-section"', fallback)
        failed = "".join(bench.render_self_review_html({"status": "failed", "reason": "timeout"}))
        self.assertNotIn('class="review-rating"', failed)
        self.assertIn("生成失败", failed)

    def test_charts_precede_tables_and_convert_latency_without_mutation(self):
        with self.completed(["html"]) as (_, summary, _):
            before = copy.deepcopy(summary)
            document = bench.render_html_report(summary)
            for heading in ("双模式对照", "关闭思考 · 性能明细", "开启思考 · 性能明细"):
                start = document.index(">" + heading + "</h2>")
                self.assertLess(document.index("<figure", start), document.index("<table", start))
            row = summary["cells"][0]
            self.assertEqual(
                bench.chart_point(row, "ttfo", "p50")["value"],
                row["metrics"]["latency_ms"]["ttfo"]["p50"] / 1000,
            )
            self.assertEqual(summary, before)

    def test_chart_null_gaps_zero_values_and_escaping(self):
        import xml.etree.ElementTree as ET

        def point(value):
            return {"value": value, "flagged": True, "source": '<bad"id>', "note": "<script>"}

        series = [("<mode>", [(128, point(0)), (256, point(None)), (512, point(10))])]
        document = bench.html_line_chart("<title>", series)
        root = ET.fromstring(document[document.index("<svg") : document.index("</svg>") + 6])
        ns = {"s": "http://www.w3.org/2000/svg"}
        circles = root.findall("s:circle", ns)
        self.assertEqual(len(circles), 2)  # Keep real zero; do not fabricate a missing point.
        self.assertEqual(circles[0].attrib["cy"], "200.00")
        self.assertEqual(circles[0].attrib["fill"], "white")
        line = next(p for p in root.findall("s:path", ns) if p.attrib.get("d", "").count("M") == 2)
        self.assertNotIn("L", line.attrib["d"])  # Never connect across the null.
        self.assertNotIn("<script>", document)
        matrix = bench.html_matrix_chart(
            "value", ["zero", "missing"], [("mode", [point(0), point(None)])]
        )
        self.assertIn("0.00*", matrix)
        self.assertIn("—", matrix)

    def test_key_scene_uses_lowest_shared_load_without_filtering_failures(self):
        rows = [
            {
                "mode": mode,
                "input_characters": size,
                "concurrency": concurrency,
                "status": "partial" if (size, concurrency) == (256, 1) else "completed",
            }
            for size, concurrency, modes in [
                (128, 1, ["off"]),
                (256, 3, bench.MODES),
                (256, 1, bench.MODES),
            ]
            for mode in modes
        ]
        selected = bench.key_scene_rows(list(reversed(rows)))
        self.assertEqual({r["input_characters"] for r in selected.values()}, {256})
        self.assertEqual({r["concurrency"] for r in selected.values()}, {1})
        self.assertEqual(selected["on"]["status"], "partial")
        self.assertIsNone(bench.key_scene_rows([r for r in rows if r["mode"] == "off"]))
        self.assertEqual(bench.key_scene_html([]), "")

    def test_anomaly_links_match_real_rows_and_keep_raw_summary_unchanged(self):
        from html.parser import HTMLParser

        class Links(HTMLParser):
            def __init__(self):
                super().__init__()
                self.ids, self.targets, self.marks = [], [], []

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if "id" in attrs:
                    self.ids.append(attrs["id"])
                if attrs.get("href", "").startswith("#"):
                    self.targets.append(attrs["href"][1:])
                if "data-cell-id" in attrs:
                    self.marks.append(attrs["data-cell-id"])

        with self.completed(["md"]) as (_, summary, _):
            originals = summary["cells"]
            summary["cells"] = []
            summary["config"]["input_characters"] = [128, 256, 512]
            for original in originals:
                for size, value in ((128, 100), (256, 150), (512, 90)):
                    row = copy.deepcopy(original)
                    row.update(input_characters=size, id=f"{original['mode']}-{size}<unsafe>")
                    row["metrics"]["latency_ms"]["ttft"]["p95"] = value
                    row["metrics"]["latency_ms"]["e2e"]["p95"] = 100
                    row["metrics"]["aggregate_output_tps"] = 100
                    summary["cells"].append(row)
            before = copy.deepcopy(summary)
            highlights = bench.performance_highlights(summary)
            self.assertEqual(len(highlights), 2)
            document = bench.render_html_report(summary)
            parsed = Links()
            parsed.feed(document)
            self.assertEqual(len(parsed.ids), len(set(parsed.ids)))
            self.assertTrue(parsed.targets)
            self.assertFalse(set(parsed.targets) - set(parsed.ids))
            for item in highlights:
                self.assertEqual(item["current"]["input_characters"], 256)
                self.assertIn(item["current"]["id"], parsed.marks)
                self.assertIn(bench.detail_anchor(item["current"]["id"]), parsed.targets)
            self.assertIn("TTFT P95（秒", document)
            self.assertIn("升高 50.0%", document)
            self.assertIn("局部劣化", document)
            self.assertIn('class="focus-row"', document)
            self.assertEqual(document.count('class="scene-card"'), 2)
            self.assertLess(document.index('class="key-scene"'), document.index("性能变化分析"))
            self.assertEqual(document.count(bench.CHART_NOTE), 1)
            self.assertNotIn("浅 → 深", document)
            self.assertNotIn("<unsafe>", document)
            self.assertEqual(summary, before)

    def test_large_matrix_highlights_are_bounded_without_mutating_metrics(self):
        with self.completed(["md"], thinking_modes=["off"]) as (_, summary, _):
            original = summary["cells"][0]
            summary["cells"] = []
            for concurrency in (1, 3, 5):
                for size in (128, 256, 512, 1024, 2048):
                    row = copy.deepcopy(original)
                    row.update(input_characters=size, concurrency=concurrency)
                    for metric in ("ttft", "e2e"):
                        row["metrics"]["latency_ms"][metric]["p95"] = size * concurrency
                    summary["cells"].append(row)
            before = copy.deepcopy(summary)
            findings = bench.performance_observations(summary)
            self.assertGreater(len(findings), 0)
            self.assertLessEqual(len(findings), 3)
            self.assertLess(sum(map(len, findings)), 700)
            self.assertEqual(summary, before)


if __name__ == "__main__":
    unittest.main()
