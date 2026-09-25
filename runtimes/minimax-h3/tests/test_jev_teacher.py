"""Teacher dataset parse/export tests. CPU only. No Jev SDK, no GPU."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = Path(__file__).resolve().parents[1]
NODE = RUNTIME / "ComfyUI" / "custom_nodes" / "ComfyUI-MiniMax-H3-009jev"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "jev"
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, str(NODE))

import jev_teacher  # noqa: E402
import native_sla_policy  # noqa: E402


class NativeSlaPolicyTests(unittest.TestCase):
    def test_const_never_uses_jev_worker(self):
        for name, keep in native_sla_policy.CONST_POLICIES.items():
            self.assertFalse(native_sla_policy.uses_jev_worker(name), name)
            self.assertEqual(native_sla_policy.initial_keeps(name), [keep] * 50)

    def test_jev_first_and_author_fixed_still_call_worker(self):
        for name in ("jev_first", "fixed5", "fixed10"):
            self.assertTrue(native_sla_policy.uses_jev_worker(name), name)

    def test_table_policy_never_uses_jev_and_logs_per_step_keeps(self):
        table = [[10] * 50 for _ in range(4)]
        table[0][7] = 10
        table[1][7] = 5
        table[2][7] = 3
        table[3][7] = 1
        table[1][0] = 10  # forced back to 5
        self.assertFalse(native_sla_policy.uses_jev_worker("table"))
        norm = native_sla_policy.normalize_keep_table(table)
        self.assertEqual(norm[1][0], 5.0)
        self.assertEqual(norm[2][7], 3.0)
        events = native_sla_policy.simulate_keep_schedule("table", table)
        self.assertEqual(events[0]["reason"], "table")
        self.assertNotEqual(events[0]["reason"], "jev")
        keeps_by_step = [e["applied_layer_keep_percent"] for e in events if e["event"] == "step"]
        self.assertEqual(len(keeps_by_step), 4)
        self.assertEqual(keeps_by_step[0][7], 10.0)
        self.assertEqual(keeps_by_step[1][7], 5.0)
        self.assertEqual(keeps_by_step[2][7], 3.0)
        self.assertEqual(keeps_by_step[3][7], 1.0)
        for step_keeps in keeps_by_step[1:]:
            self.assertEqual(step_keeps[0], 5.0)
        const10 = native_sla_policy.simulate_keep_schedule("const10")
        const_keeps = [e["applied_layer_keep_percent"] for e in const10 if e["event"] == "step"]
        self.assertTrue(all(k == 10.0 for row in const_keeps for k in row))
        self.assertNotEqual(keeps_by_step, const_keeps)

    def test_table_policy_accepts_n_by_50(self):
        for n in (15, 20, 32):
            table = [[10] * 50 for _ in range(n)]
            table[n - 1][7] = 3
            norm = native_sla_policy.normalize_keep_table(table)
            self.assertEqual(len(norm), n)
            events = native_sla_policy.simulate_keep_schedule("table", table)
            keeps = [e["applied_layer_keep_percent"] for e in events if e["event"] == "step"]
            self.assertEqual(len(keeps), n)
            self.assertEqual(keeps[-1][7], 3.0)
            self.assertEqual(keeps[1][0], 5.0)

    def test_nstep_teacher_harvest_yields_n_times_50(self):
        n = 20
        probs = {"1": 0.0, "3": 0.0, "5": 0.0, "10": 1.0}
        dec50 = {
            str(b): {"choice": 10, "confidence": 0.91, "probabilities": probs}
            for b in range(50)
        }
        dec49 = {
            str(b): {"choice": 10, "confidence": 0.91, "probabilities": probs}
            for b in range(1, 50)
        }
        blocks = {
            str(b): {
                "audio": {"residual_relative_l2": 0.1, "rank": 0.5, "cross_step_change": None},
                "video": {"residual_relative_l2": 0.2, "rank": 0.4, "cross_step_change": None},
            }
            for b in range(50)
        }
        events = [
            {
                "event": "initial_decision",
                "reason": "jev",
                "initial_policy": "jev_first",
                "applied_layer_keep_percent": [10.0] * 50,
                "answer": {"decisions": dec50, "model": "jev-1.13.0"},
            },
            {
                "event": "begin",
                "n_steps": n,
                "sampler": "sample_euler",
                "initial_policy": "jev_first",
            },
        ]
        for step in range(1, n + 1):
            applied = [10.0] * 50 if step == 1 else [5.0] + [10.0] * 49
            ev = {
                "event": "step",
                "step": step,
                "n_steps": n,
                "applied_layer_keep_percent": applied,
                "actual_attention": {str(b): "sla" for b in range(50)},
            }
            if step < n:
                ev["answer"] = {"decisions": dec49, "model": "jev-1.13.0"}
                ev["state"] = {"blocks": blocks}
            events.append(ev)
        events.append({"event": "end", "requests": n})
        gen, rows, calls = jev_teacher.records_from_events(
            events, generation_id="hq20", prompt="fox", seed=1
        )
        self.assertEqual(gen["steps"], 20)
        self.assertEqual(len(rows), 20 * 50)
        self.assertEqual({r["step"] for r in rows}, set(range(1, 21)))
        self.assertEqual({r["layer"] for r in rows}, set(range(50)))
        self.assertEqual(len(calls), 20)
        self.assertEqual(rows[0]["num_steps"], 20)
        errs = jev_teacher.validate_layer_rows(rows, generation_id="hq20")
        self.assertEqual(errs, [], errs)
        later0 = [r for r in rows if r["step"] > 1 and r["layer"] == 0]
        self.assertEqual(len(later0), 19)
        for row in later0:
            self.assertEqual(row["applied_keep"], 5)
            self.assertEqual(row["fallback_reason"], "block0_later_keep5")

    def test_confidence_floor_first_vs_later(self):
        applied, triggered, reason = native_sla_policy.apply_confidence_floor(
            1.0, 0.19, first_step=False
        )
        self.assertEqual(applied, 5.0)
        self.assertTrue(triggered)
        self.assertEqual(reason, "confidence_floor")
        applied, triggered, reason = native_sla_policy.apply_confidence_floor(
            1.0, 0.19, first_step=True
        )
        self.assertEqual(applied, 10.0)
        self.assertTrue(triggered)
        applied, triggered, reason = native_sla_policy.apply_confidence_floor(
            3.0, 0.77, first_step=False
        )
        self.assertEqual(applied, 3.0)
        self.assertFalse(triggered)
        self.assertIsNone(reason)


class BenchmarkConfigTests(unittest.TestCase):
    def test_case_json_builds_matched_argv(self):
        case = json.loads((ROOT / "docs" / "jev-baseline" / "case.json").read_text(encoding="utf-8"))
        cmds = jev_teacher.benchmark_commands(case)
        self.assertEqual([c["policy"] for c in cmds], case["policies"])
        shared = None
        for row in cmds:
            argv = row["argv"]
            self.assertEqual(argv[0:4], ["gemmy", "video", "h3", "generate"])
            self.assertIn("--no-compile-ir", argv)
            self.assertEqual(argv[argv.index("--seed") + 1], "42")
            self.assertEqual(argv[argv.index("--width") + 1], "864")
            self.assertEqual(argv[argv.index("--height") + 1], "864")
            self.assertEqual(argv[argv.index("--steps") + 1], "4")
            self.assertEqual(argv[argv.index("--first-frame") + 1], case["first_frame"])
            self.assertEqual(argv[argv.index("--prompt-file") + 1], case["prompt_file"])
            core = [x for x in argv if x not in ("--jev", "--sla-fixed", "--no-sla", "1", "3", "5", "10")]
            if shared is None:
                shared = core
            else:
                self.assertEqual(core, shared)
        policies = {c["policy"]: c["argv"] for c in cmds}
        self.assertIn("--jev", policies["jev"])
        self.assertNotIn("--sla-fixed", policies["jev"])
        self.assertNotIn("--no-sla", policies["jev"])
        self.assertEqual(policies["fixed-5"][policies["fixed-5"].index("--sla-fixed") + 1], "5")
        self.assertNotIn("--jev", policies["fixed-5"])
        self.assertIn("--no-sla", policies["no-sla"])
        self.assertNotIn("--jev", policies["no-sla"])
        self.assertNotIn("--sla-fixed", policies["no-sla"])


class BenchLogScanTests(unittest.TestCase):
    def test_parse_log_metrics_flags_sla_vs_dense(self):
        import jev_baseline_bench as bench

        dense = (
            "[h3 generate] no-sla 4-step res_multistep (no H3JevNativeSLAPatch)\n"
            "SamplerCustomAdvanced 1/4 [00:50<00:00, 12.50s/it]\n"
            "[h3-comfy] wrote x in 180.0s\n"
        )
        sla = (
            dense.replace("no H3JevNativeSLAPatch", "H3JevNativeSLAPatch")
            + '[009jev] {"event":"begin","initial_policy":"const10"}\n'
            + "H3JevNativeSLAPatch\n"
            + "h3_sparse_attention SparseAttnPatch\n"
        )
        d = bench.parse_log_metrics(dense)
        s = bench.parse_log_metrics(sla)
        self.assertFalse(d["log_has_009jev"])
        self.assertFalse(d["log_has_H3JevNativeSLAPatch"])
        self.assertFalse(d["log_has_h3_sparse_attention"])
        self.assertEqual(d["sampler_seconds"], 50)
        self.assertTrue(s["log_has_009jev"])
        self.assertTrue(s["log_has_H3JevNativeSLAPatch"])
        self.assertTrue(s["log_has_h3_sparse_attention"])


class TeacherParseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        events_path = FIXTURES / "h3_jev_i2va_events.json"
        if not events_path.is_file():
            raise unittest.SkipTest(f"missing fixture {events_path}")
        cls.events = jev_teacher.load_events(events_path)
        prompt = ""
        for event in cls.events:
            state = event.get("state") or {}
            if state.get("prompt"):
                prompt = str(state["prompt"])
                break
        cls.prompt = prompt or "fixture prompt"
        cls.gen, cls.rows, cls.calls = jev_teacher.records_from_events(
            cls.events,
            generation_id="h3_jev_i2va",
            prompt=prompt,
            seed=42,
            quality_tag="clean",
        )

    def test_fifty_layers_four_steps(self):
        self.assertEqual(len(self.rows), 200)
        self.assertEqual({r["layer"] for r in self.rows}, set(range(50)))
        self.assertEqual({r["step"] for r in self.rows}, {1, 2, 3, 4})
        errs = jev_teacher.validate_layer_rows(self.rows, generation_id="h3_jev_i2va")
        self.assertEqual(errs, [], errs)

    def test_choices_only_valid(self):
        for row in self.rows:
            self.assertIn(row["applied_keep"], {1, 3, 5, 10})
            if row["jev_raw_choice"] is not None:
                self.assertIn(row["jev_raw_choice"], {1, 3, 5, 10})

    def test_block0_first_step_only_5_or_10(self):
        row = next(r for r in self.rows if r["step"] == 1 and r["layer"] == 0)
        self.assertIn(row["jev_raw_choice"], (5, 10))
        self.assertIn(row["applied_keep"], (5, 10))
        later = [r for r in self.rows if r["step"] > 1 and r["layer"] == 0]
        self.assertTrue(later)
        for row in later:
            self.assertEqual(row["applied_keep"], 5)
            self.assertIsNone(row["jev_raw_choice"])
            self.assertEqual(row["fallback_reason"], "block0_later_keep5")
            self.assertFalse(row["fallback_triggered"])

    def test_probabilities_and_confidence_present(self):
        jev_rows = [r for r in self.rows if r["jev_raw_choice"] is not None]
        self.assertGreater(len(jev_rows), 50)
        for row in jev_rows:
            self.assertIsNotNone(row["jev_confidence"])
            self.assertTrue(0.0 <= row["jev_confidence"] <= 1.0)
            probs = row["jev_probabilities"]
            self.assertEqual(set(probs), {"1", "3", "5", "10"})
            self.assertAlmostEqual(sum(probs.values()), 1.0, delta=0.05)

    def test_confidence_fallback_raw_1_applied_5(self):
        hits = [
            r
            for r in self.rows
            if r["jev_raw_choice"] == 1
            and abs(float(r["jev_confidence"]) - 0.19) < 1e-9
            and r["applied_keep"] == 5
            and r["fallback_triggered"] is True
        ]
        self.assertTrue(hits, "expected the observed layer-4 confidence floor")
        row = hits[0]
        self.assertEqual(row["fallback_reason"], "confidence_floor")
        self.assertNotEqual(row["jev_raw_choice"], row["applied_keep"])
        self.assertEqual(row["generation_id"], "h3_jev_i2va")

    def test_applied_keep_matches_step_event(self):
        step2 = next(e for e in self.events if e.get("event") == "step" and e.get("step") == 2)
        applied = [int(round(x)) for x in step2["applied_layer_keep_percent"]]
        got = [r["applied_keep"] for r in self.rows if r["step"] == 2]
        self.assertEqual(got, applied)

    def test_no_secrets_in_export(self):
        blob = json.dumps({"generation": self.gen, "rows": self.rows})
        self.assertEqual(jev_teacher.scan_secrets(blob), [])
        self.assertNotIn("TYPESAFE_API_KEY=", blob)
        poisoned = dict(self.rows[0])
        poisoned["prompt"] = "TYPESAFE_API_KEY=sk-thisisafakekeyvaluexx"
        self.assertTrue(jev_teacher.scan_secrets(poisoned))

    def test_raw_not_collapsed_into_applied(self):
        diffs = [r for r in self.rows if r["jev_raw_choice"] not in (None, r["applied_keep"])]
        self.assertTrue(diffs)
        for row in diffs:
            self.assertTrue(row["fallback_triggered"])

    def test_append_jsonl_gains_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            synthetic = json.loads(json.dumps(self.events))
            summary = jev_teacher.append_from_run(
                dataset_dir=dataset,
                events=synthetic,
                generation={
                    "generation_id": "synthetic-append",
                    "prompt": self.prompt,
                    "seed": 42,
                    "quality_tag": "clean",
                    "width": 864,
                    "height": 864,
                },
            )
            self.assertEqual(summary["layer_rows"], 200)
            gens = jev_teacher.read_jsonl(dataset / "generations.jsonl")
            layers = jev_teacher.read_jsonl(dataset / "layer_decisions.jsonl")
            self.assertEqual(len(gens), 1)
            self.assertEqual(gens[0]["generation_id"], "synthetic-append")
            self.assertEqual({r["generation_id"] for r in layers}, {"synthetic-append"})
            self.assertEqual(len(layers), 200)

    def test_smeared_log_fixture_if_present(self):
        path = FIXTURES / "h3_jev_009_events.json"
        if not path.is_file():
            self.skipTest("009 events fixture not copied")
        events = jev_teacher.load_events(path)
        gen, rows, _ = jev_teacher.records_from_events(
            events, generation_id="h3_jev_009", prompt="smeared", seed=42, quality_tag="smeared"
        )
        self.assertEqual(gen["quality_tag"], "smeared")
        self.assertEqual({r["quality_tag"] for r in rows}, {"smeared"})
        self.assertEqual(len(rows), 200)


if __name__ == "__main__":
    unittest.main()
