"""Agent workload regression tests; local deterministic HTTP fixtures only."""

import contextlib
import copy
import hashlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from test_standalone_llm_benchmark import (
    SECRET,
    LocalServer,
    bench,
    config_for,
    enabled,
    event,
    start_stream,
)


def agent_config(url="http://127.0.0.1:1/v1/chat/completions", scenarios=None):
    config = config_for(url)
    config.update(input_characters=[128], concurrency=[1, 2], thinking_modes=["off"], repetitions=1)
    config["agent_performance"] = {
        "enabled": True,
        "scenarios": scenarios or ["long_context", "loop"],
        "concurrency": [1, 2],
        "repetitions": 1,
        "long_context": {"input_characters": [128, 256]},
        "loop": {
            "initial_input_characters": 128,
            "model_calls_per_session": 3,
            "tool_result_characters": 128,
            "tool_delay_ms": 5,
        },
    }
    return bench.validate_config(config)


def agent_response(handler, body, *, usage=True, bad=False):
    start_stream(handler)
    is_tool = isinstance(body.get("tool_choice"), dict)
    if enabled(body):
        event(handler, {"choices": [{"delta": {"reasoning_content": "private-agent-reasoning"}}]})
        time.sleep(0.004)
    if is_tool:
        for delta in (
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "benchmark_", "arguments": "{"},
            },
            {"index": 0, "function": {"name": "step", "arguments": '"bad":true}' if bad else "}"}},
        ):
            event(handler, {"choices": [{"delta": {"tool_calls": [delta]}}]})
            time.sleep(0.003)
        reason = "tool_calls"
    else:
        event(handler, {"choices": [{"delta": {"content": "private-agent-answer"}}]})
        time.sleep(0.005)
        event(handler, {"choices": [{"delta": {"content": " done"}}]})
        reason = "stop"
    last = {"choices": [{"delta": {}, "finish_reason": reason}]}
    if usage:
        last["usage"] = {
            "prompt_tokens": 50 * len(body["messages"]),
            "completion_tokens": 12,
            "completion_tokens_details": {"reasoning_tokens": 3 if enabled(body) else 0},
        }
    event(handler, last)
    event(handler, "[DONE]")


def run(config, directory):
    with contextlib.redirect_stdout(io.StringIO()):
        return bench.run_benchmark(config, directory, SECRET)


class AgentConfigurationTests(unittest.TestCase):
    def test_disabled_and_missing_keep_original_plan(self):
        config = config_for()
        self.assertIsNone(bench.make_agent_plan(config))
        config["agent_performance"] = {"enabled": False}
        self.assertIsNone(bench.make_agent_plan(bench.validate_config(config)))

    def test_default_plan_reuses_only_matching_baseline(self):
        config = bench.validate_config(
            {
                "schema_version": bench.CONFIG_VERSION,
                "model": {"name": "Qwen3", "api_url": "http://localhost:1/v1/chat/completions"},
                "agent_performance": {"enabled": True},
            }
        )
        plan = bench.make_agent_plan(config)
        self.assertEqual(plan["referenced_requests"], 288)
        self.assertEqual(plan["additional_performance_requests_max"], 1056)
        self.assertEqual(plan["additional_sessions_max"], 96)
        self.assertEqual(plan["performance_tool_calls_max"], 864)
        config["agent_performance"]["repetitions"] = 4
        self.assertEqual(bench.make_agent_plan(config)["referenced_requests"], 0)

    def test_invalid_agent_configuration(self):
        for fragment in (
            {"enabled": "true"},
            {"enabled": True, "scenarios": []},
            {"enabled": True, "concurrency": [0]},
            {"enabled": True, "scenarios": ["image_input"]},
            {"enabled": True, "loop": {"model_calls_per_session": 1}},
            {"enabled": True, "targets": {"loop": {"min_success_rate": 2}}},
        ):
            config = config_for()
            config["agent_performance"] = fragment
            with self.subTest(fragment=fragment), self.assertRaises(bench.BenchmarkError):
                bench.validate_config(config)

    def test_tool_delta_is_not_ready_until_finished(self):
        obs = bench.ToolObservation()
        obs.add(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "function": {"name": "benchmark_step", "arguments": "{}"},
                    }
                ]
            },
            0.1,
            None,
        )
        self.assertFalse(obs.validated())
        obs.add({}, 0.3, "tool_calls")
        self.assertTrue(obs.validated())
        self.assertEqual(obs.finished, 0.3)
        with self.assertRaises(bench.BenchmarkError):
            obs.add({"tool_calls": [{"index": 0, "function": {"arguments": "x"}}]}, 0.4, None)


class AgentConclusionTests(unittest.TestCase):
    def rows(self):
        rows = []
        for size, latency in ((128, 100), (256, 150), (512, 90)):
            metrics = bench.calculate_metrics([], 0)
            metrics.update(
                success_count=3,
                attempted_count=3,
                success_rate=1,
                aggregate_output_tps=100,
                truncated_count=0,
            )
            for key in ("ttft", "ttfo", "e2e"):
                metrics["latency_ms"][key] = bench.distribution([latency] * 3)
            rows.append(
                dict(
                    id=str(size),
                    scenario="long_context",
                    mode="off",
                    input_characters=size,
                    concurrency=1,
                    status="completed",
                    reason=None,
                    metrics=metrics,
                    rounds=[],
                    max_tokens=512,
                )
            )
        return rows

    def test_degradation_reports_numeric_transition_and_recovery(self):
        rows = self.rows()
        findings = bench.agent_observations({"cells": rows}, agent_config())
        result = next(x for x in findings if x["kind"] == "degradation")
        self.assertEqual(result["cell_ids"], ["128", "256"])
        self.assertEqual(result["evidence"]["relative_change"], 0.5)
        self.assertEqual(result["evidence"]["next_value"], 90)
        self.assertIn("128 → 256", result["text"])
        self.assertIn("局部劣化", result["text"])

    def test_sparse_metric_and_truncation_cannot_support_degradation(self):
        rows = self.rows()
        for row in rows:
            for key in ("ttft", "ttfo", "e2e"):
                row["metrics"]["latency_ms"][key]["count"] = 1
        findings = bench.agent_observations({"cells": rows}, agent_config())
        self.assertFalse(any(x["kind"] == "degradation" for x in findings))
        rows = self.rows()
        rows[1]["metrics"]["truncated_count"] = 1
        findings = bench.agent_observations({"cells": rows}, agent_config())
        self.assertFalse(any(x["kind"] == "degradation" for x in findings))
        self.assertTrue(any(x["kind"] == "truncated" for x in findings))


class AgentRunTests(unittest.TestCase):
    def test_loop_baseline_report_and_offline_rebuild(self):
        with tempfile.TemporaryDirectory() as temp, LocalServer(agent_response) as server:
            config = agent_config(server.url)
            directory = Path(temp) / "evidence"
            summary = run(config, directory)
            agent = summary["agent_performance"]
            self.assertEqual(agent["status"], "completed")
            self.assertEqual(agent["plan"]["referenced_requests"], 3)
            self.assertEqual(agent["plan"]["additional_performance_requests_max"], 12)
            self.assertEqual(len(server.requests), 19)  # 1+3 baseline; 1+2 preparation; 12 extra
            rows = [x for x in agent["cells"] if x["scenario"] == "loop"]
            self.assertTrue(all(x["metrics"]["session_completion_rate"] == 1 for x in rows))
            self.assertEqual(rows[1]["metrics"]["session_completed"], 2)
            self.assertEqual(rows[1]["metrics"]["session_tokens"]["prompt_tokens"]["p50"], 600)
            self.assertEqual(rows[1]["metrics"]["session_tokens"]["completion_tokens"]["p50"], 36)
            self.assertTrue(
                all(x["metrics"]["output_tps"]["count"] == 0 for x in rows[1]["rounds"][:-1])
            )
            self.assertGreater(
                rows[1]["rounds"][1]["metrics"]["tokens"]["prompt_tokens"]["p50"],
                rows[1]["rounds"][0]["metrics"]["tokens"]["prompt_tokens"]["p50"],
            )
            self.assertEqual(summary["planned_performance_requests"], 3)
            md = (directory / "report.md").read_text()
            html = (directory / "report.html").read_text()
            for text in (md, html):
                self.assertIn("Agent 能力评估", text)
                self.assertIn("常规基线引用", text)
                self.assertLess(text.index("Agent 能力评估"), text.index("测量口径与附录"))
            self.assertIn("<svg", html)
            files = "\n".join(p.read_text() for p in directory.iterdir() if p.is_file())
            for secret in (
                SECRET,
                "private-agent-answer",
                "private-agent-reasoning",
                '"arguments": "{}"',
            ):
                self.assertNotIn(secret, files)
            with mock.patch.object(bench, "perform_request", side_effect=AssertionError("offline")):
                rebuilt = bench.generate_report(directory)
            self.assertEqual(rebuilt, summary)
            self.assertEqual((directory / "report.md").read_text(), md)
            self.assertEqual((directory / "report.html").read_text(), html)

    def test_missing_usage_does_not_fabricate_tps(self):
        def behavior(handler, body):
            agent_response(handler, body, usage="tools" not in body)

        with tempfile.TemporaryDirectory() as temp, LocalServer(behavior) as server:
            summary = run(agent_config(server.url, ["loop"]), Path(temp) / "e")
            for row in summary["agent_performance"]["cells"]:
                self.assertIsNone(row["metrics"]["aggregate_output_tps"])
                self.assertEqual(row["metrics"]["usage_coverage"], 0)

    def test_invalid_tool_stops_only_agent_scenario(self):
        def behavior(handler, body):
            agent_response(handler, body, bad=True)

        with tempfile.TemporaryDirectory() as temp, LocalServer(behavior) as server:
            summary = run(agent_config(server.url), Path(temp) / "e")
            self.assertTrue(all(x["status"] == "completed" for x in summary["cells"]))
            rows = summary["agent_performance"]["cells"]
            self.assertTrue(all(x["status"] == "skipped" for x in rows if x["scenario"] == "loop"))
            self.assertTrue(
                all(x["status"] == "completed" for x in rows if x["scenario"] == "long_context")
            )
            self.assertEqual(summary["status"], "partial")

    def test_partial_journal_preserves_unresolved_and_no_throughput(self):
        with tempfile.TemporaryDirectory() as temp, LocalServer(agent_response) as server:
            directory = Path(temp) / "e"
            run(agent_config(server.url, ["loop"]), directory)
            path = directory / "agent_requests.jsonl"
            records = [json.loads(x) for x in path.read_text().splitlines()]
            last_request = next(
                i
                for i in range(len(records) - 1, -1, -1)
                if records[i]["record_type"] == "scheduled"
            )
            path.write_text(
                "\n".join(json.dumps(x) for x in records[: last_request + 1]) + "\n" + '{"broken":'
            )
            summary = bench.generate_report(directory)
            agent = summary["agent_performance"]
            self.assertGreater(agent["pending_requests"], 0)
            self.assertGreater(agent["pending_sessions"], 0)
            self.assertEqual(agent["status"], "partial")
            self.assertIsNone(agent["cells"][-1]["metrics"]["aggregate_output_tps"])

    def test_cancel_preserves_baseline_and_specialty(self):
        controller = bench.StopController()

        def behavior(handler, body):
            agent_response(handler, body)
            if "tools" in body:
                controller.cancel()

        with tempfile.TemporaryDirectory() as temp, LocalServer(behavior) as server:
            with contextlib.redirect_stdout(io.StringIO()):
                summary = bench.run_benchmark(
                    agent_config(server.url, ["loop"]), Path(temp) / "e", SECRET, controller
                )
            self.assertEqual(summary["status"], "cancelled")
            self.assertTrue(all(x["metrics"]["success_rate"] == 1 for x in summary["cells"]))
            self.assertEqual(summary["agent_performance"]["status"], "partial")

    def test_disabled_does_not_add_network_or_report(self):
        with tempfile.TemporaryDirectory() as temp, LocalServer(agent_response) as server:
            config = agent_config(server.url)
            config["agent_performance"] = {"enabled": False}
            directory = Path(temp) / "e"
            summary = run(config, directory)
            self.assertEqual(len(server.requests), 4)
            self.assertNotIn("agent_performance", summary)
            self.assertFalse((directory / "agent_requests.jsonl").exists())
            self.assertNotIn("Agent 能力评估", (directory / "report.html").read_text())

    def test_thinking_history_is_transient_and_replayed(self):
        with tempfile.TemporaryDirectory() as temp, LocalServer(agent_response) as server:
            config = agent_config(server.url, ["loop"])
            config["thinking_modes"] = ["on"]
            directory = Path(temp) / "e"
            summary = run(config, directory)
            self.assertEqual(summary["agent_performance"]["status"], "completed")
            continuing = [
                body for body, _ in server.requests if "tools" in body and len(body["messages"]) > 2
            ]
            self.assertTrue(continuing)
            self.assertTrue(
                any(
                    m.get("reasoning_content") == "private-agent-reasoning"
                    for m in continuing[0]["messages"]
                )
            )
            self.assertNotIn(
                "private-agent-reasoning", (directory / "agent_requests.jsonl").read_text()
            )

    def test_loop_window_includes_tool_delay(self):
        with tempfile.TemporaryDirectory() as temp, LocalServer(agent_response) as server:
            config = agent_config(server.url, ["loop"])
            config["agent_performance"]["loop"]["tool_delay_ms"] = 50
            summary = run(config, Path(temp) / "e")
            row = summary["agent_performance"]["cells"][0]
            self.assertGreater(row["metrics"]["session_window_seconds"], 0.1)
            self.assertAlmostEqual(
                row["metrics"]["aggregate_output_tps"],
                36 / row["metrics"]["session_window_seconds"],
            )

    def test_identity_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temp, LocalServer(agent_response) as server:
            directory = Path(temp) / "e"
            run(agent_config(server.url, ["loop"]), directory)
            path = directory / "agent_requests.jsonl"
            records = [json.loads(x) for x in path.read_text().splitlines()]
            item = next(x for x in records if x["record_type"] == "result")
            item["request"]["turn"] = 99
            path.write_text("\n".join(json.dumps(x) for x in records) + "\n")
            with self.assertRaises(bench.BenchmarkError):
                bench.generate_report(directory)

    def test_targets_and_html_escaping(self):
        with tempfile.TemporaryDirectory() as temp, LocalServer(agent_response) as server:
            config = agent_config(server.url, ["loop"])
            config["agent_performance"]["targets"] = {
                "loop": {"min_session_completion_rate": 1, "session_p95_ms": 1}
            }
            summary = run(config, Path(temp) / "e")
            checks = summary["agent_performance"]["cells"][0]["target_checks"]
            self.assertEqual([x["result"] for x in checks], ["本次观测达标", "本次观测未达标"])
            changed = copy.deepcopy(summary["agent_performance"])
            changed["observations"][0]["text"] = '<img src=x onerror="alert(1)">'
            html = "".join(bench.render_agent_html(changed))
            self.assertNotIn("<img", html)
            self.assertIn("&lt;img", html)


class AgentMediaAndBoundaryTests(unittest.TestCase):
    def test_image_audio_metadata_and_offline_without_files(self):
        import struct
        import wave
        import zlib

        with tempfile.TemporaryDirectory() as temp, LocalServer(agent_response) as server:
            root = Path(temp)

            def chunk(kind, body):
                return (
                    struct.pack(">I", len(body))
                    + kind
                    + body
                    + struct.pack(">I", zlib.crc32(kind + body))
                )

            png = root / "private-image-name.png"
            png.write_bytes(
                b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
                + chunk(b"IEND", b"")
            )
            wav = root / "private-audio-name.wav"
            with wave.open(str(wav), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(b"\x00\x00" * 1600)
            config = agent_config(server.url, ["loop"])
            config["agent_performance"]["scenarios"] = ["image_input", "audio_input"]
            config["agent_performance"]["media_samples"] = [
                {"kind": "image", "path": str(png), "count": 2},
                {"kind": "audio", "path": str(wav)},
            ]
            directory = root / "e"
            summary = run(config, directory)
            self.assertEqual(summary["agent_performance"]["status"], "completed")
            self.assertEqual(len(server.requests), 12)
            media_requests = [
                body
                for body, _ in server.requests
                if isinstance(body["messages"][0]["content"], list)
            ]
            self.assertEqual(len(media_requests), 8)
            meta = summary["config"]["agent_performance"]["media_samples"]
            self.assertEqual(meta[0]["count"], 2)
            self.assertEqual(meta[1]["duration_ms"], 100)
            self.assertEqual(meta[0]["sha256"], hashlib.sha256(png.read_bytes()).hexdigest())
            png.unlink()
            wav.unlink()
            self.assertEqual(bench.generate_report(directory), summary)
            saved = "\n".join(p.read_text() for p in directory.iterdir())
            self.assertNotIn("private-image-name", saved)
            self.assertNotIn("private-audio-name", saved)
            self.assertNotIn("base64", saved)

    def test_invalid_media_fails_before_network(self):
        with tempfile.TemporaryDirectory() as temp, LocalServer(agent_response) as server:
            config = agent_config(server.url)
            config["agent_performance"]["scenarios"] = ["image_input"]
            config["agent_performance"]["media_samples"] = [
                {"kind": "image", "path": str(Path(temp) / "missing.png")}
            ]
            with self.assertRaises(bench.BenchmarkError):
                run(config, Path(temp) / "e")
            self.assertFalse(server.requests)

    def test_session_timeout_covers_tool_wait(self):
        def behavior(handler, body):
            if "tools" not in body:
                time.sleep(0.08)  # A Loop deadline must not constrain single-request scenarios.
            agent_response(handler, body)

        with tempfile.TemporaryDirectory() as temp, LocalServer(behavior) as server:
            config = agent_config(server.url)
            config["agent_performance"]["loop"].update(
                session_timeout_seconds=0.06, tool_delay_ms=100
            )
            summary = run(config, Path(temp) / "e")
            self.assertEqual(summary["agent_performance"]["status"], "partial")
            for row in summary["agent_performance"]["cells"]:
                if row["scenario"] == "loop":
                    self.assertEqual(row["reason"], "agent_preflight_failed")
                else:
                    self.assertEqual(row["status"], "completed")

    def test_warmup_is_excluded_and_counts_match_plan(self):
        with tempfile.TemporaryDirectory() as temp, LocalServer(agent_response) as server:
            config = agent_config(server.url)
            config["warmup"] = {"enabled": True, "requests_per_length": 1}
            summary = run(config, Path(temp) / "e")
            agent = summary["agent_performance"]
            self.assertEqual(agent["plan"]["warmup_requests_max"], 4)
            self.assertEqual(
                agent["additional_attempted_requests"],
                agent["plan"]["total_additional_requests_max"],
            )
            self.assertEqual(agent["cells"][-1]["metrics"]["attempted_count"], 6)

    def test_failed_baseline_reference_is_not_rerun(self):
        def behavior(handler, body):
            if body.get("max_tokens") == 128:
                agent_response(handler, body)
            else:
                handler.send_response(503)
                handler.end_headers()

        with tempfile.TemporaryDirectory() as temp, LocalServer(behavior) as server:
            config = agent_config(server.url, ["long_context"])
            config["agent_performance"]["long_context"]["input_characters"] = [128]
            summary = run(config, Path(temp) / "e")
            self.assertEqual(len(server.requests), 2)
            self.assertEqual(summary["agent_performance"]["additional_attempted_requests"], 0)
            self.assertTrue(
                all(
                    x["source"] == "baseline_reference"
                    for x in summary["agent_performance"]["cells"]
                )
            )

    def test_mode_contradiction_stops_remaining_agent_groups(self):
        def behavior(handler, body):
            if "tools" in body:
                start_stream(handler)
                event(handler, {"choices": [{"delta": {"reasoning_content": "unexpected"}}]})
                event(
                    handler,
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "function": {
                                                "name": "benchmark_step",
                                                "arguments": "{}",
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    },
                )
                event(handler, "[DONE]")
            else:
                agent_response(handler, body)

        with tempfile.TemporaryDirectory() as temp, LocalServer(behavior) as server:
            summary = run(agent_config(server.url, ["loop"]), Path(temp) / "e")
            agent = summary["agent_performance"]
            self.assertEqual(agent["additional_attempted_requests"], 1)
            self.assertEqual(agent["cells"][-1]["reason"], "mode_contradicted")


if __name__ == "__main__":
    unittest.main()
