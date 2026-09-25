"""CPU tests for local-controller eval. Drive shipped load/filter/repr/metrics."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME))

import local_controller_eval as ev  # noqa: E402


class DatasetFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gens, cls.rows = ev.load_dataset(root=ROOT)
        cls.eligible = ev.eligible_records(cls.rows)

    def test_eligible_raw_is_four_way(self):
        self.assertGreater(len(self.eligible), 0)
        for rec in self.eligible:
            self.assertIn(rec["jev_choice"], ev.KEEP_SET, rec["sample_id"])

    def test_block0_later_excluded_from_eligible(self):
        later0 = [
            r
            for r in self.rows
            if int(r["layer"]) == 0 and int(r["step"]) > 1
        ]
        self.assertGreater(len(later0), 0)
        for rec in later0:
            self.assertTrue(ev.is_block0_later(rec), rec["sample_id"])
            self.assertFalse(ev.is_eligible(rec), rec["sample_id"])
            self.assertIsNone(rec["jev_choice"])
            self.assertEqual(rec["applied_keep"], 5)
            self.assertEqual(rec["fallback_reason"], "block0_later_keep5")
        ids = {r["sample_id"] for r in self.eligible}
        for rec in later0:
            self.assertNotIn(rec["sample_id"], ids)

    def test_confidence_floor_example_target_is_raw_1_not_applied_5(self):
        hits = [
            r
            for r in self.rows
            if r["generation_id"] == "h3_jev_i2va"
            and int(r["step"]) == 2
            and int(r["layer"]) == 4
        ]
        self.assertEqual(len(hits), 1)
        rec = hits[0]
        self.assertEqual(rec["jev_choice"], 1)
        self.assertEqual(rec["applied_keep"], 5)
        self.assertAlmostEqual(float(rec["jev_confidence"]), 0.19)
        self.assertTrue(rec["fallback_triggered"])
        self.assertEqual(rec["fallback_reason"], "confidence_floor")
        self.assertTrue(ev.is_eligible(rec))
        self.assertNotEqual(rec["jev_choice"], rec["applied_keep"])


class RepresentationLeakTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, rows = ev.load_dataset(root=ROOT)
        cls.eligible = ev.eligible_records(rows)

    def test_three_styles_contain_no_jev_label(self):
        sample = self.eligible[0]
        later = next(r for r in self.eligible if int(r["step"]) >= 2)
        first = next(r for r in self.eligible if int(r["step"]) == 1)
        for rec in (sample, later, first):
            for style in ("raw", "derived", "semantic"):
                text = ev.build_input(rec, style)
                self.assertFalse(ev.input_leaks_label(text), f"{style} {rec['sample_id']}\n{text}")
                self.assertNotIn("jev_raw_choice", text)
                self.assertNotIn("applied_keep", text.lower())
                self.assertNotIn(str(rec["jev_choice"]) + "% applied", text)

    def test_prompt_variant_still_omits_choice(self):
        rec = next(r for r in self.eligible if r.get("prompt"))
        text = ev.build_input(rec, "raw", include_prompt=True)
        self.assertFalse(ev.input_leaks_label(text))
        self.assertIn("Generation prompt:", text)


class LogoLookupTests(unittest.TestCase):
    def test_held_out_generation_never_in_train_table(self):
        _, rows = ev.load_dataset(root=ROOT)
        eligible = ev.eligible_records(rows)
        by_gen = {}
        for r in eligible:
            by_gen.setdefault(str(r["generation_id"]), []).append(r)
        self.assertGreaterEqual(len(by_gen), 2)
        hold = next(iter(by_gen))
        train = [r for gid, rs in by_gen.items() if gid != hold for r in rs]
        self.assertTrue(all(str(r["generation_id"]) != hold for r in train))
        hold_ids = {r["sample_id"] for r in by_gen[hold]}
        self.assertTrue(all(r["sample_id"] not in hold_ids for r in train))
        preds = ev.layer_step_lookup_logo(eligible)
        self.assertEqual(len(preds), len(eligible))
        self.assertTrue(all(p in ev.KEEP_SET for p in preds))

    def test_logo_changes_when_held_out_gen_has_unique_cell(self):
        """Shipped lookup must not copy the held-out generation's own labels."""
        fake = []
        for gid, keep in (("gA", 1), ("gB", 10)):
            fake.append(
                {
                    "sample_id": f"{gid}:s2:l7",
                    "generation_id": gid,
                    "step": 2,
                    "layer": 7,
                    "jev_choice": keep,
                    "layer_depth": 0.1,
                    "previous_keep": 10,
                    "audio_residual_relative_l2": None,
                    "audio_rank": None,
                    "audio_cross_step_change": None,
                    "video_residual_relative_l2": None,
                    "video_rank": None,
                    "video_cross_step_change": None,
                    "jev_confidence": 0.9,
                    "jev_probabilities": None,
                    "applied_keep": keep,
                    "fallback_triggered": False,
                    "fallback_reason": None,
                    "quality_tag": "clean",
                    "prompt": "",
                    "num_steps": 4,
                }
            )
        preds = ev.layer_step_lookup_logo(fake)
        # gA held out trains only on gB → 10; gB held out trains only on gA → 1
        self.assertEqual(preds[0], 10)
        self.assertEqual(preds[1], 1)
        self.assertNotEqual(preds[0], fake[0]["jev_choice"])


class BaselineMetricTests(unittest.TestCase):
    def test_always_10_on_real_eligible(self):
        _, rows = ev.load_dataset(root=ROOT)
        eligible = ev.eligible_records(rows)
        preds = ev.always_predict(10, eligible)
        bundle = ev.metrics_bundle(eligible, preds)
        y = [r["jev_choice"] for r in eligible]
        expected = sum(1 for t in y if t == 10) / len(y)
        self.assertAlmostEqual(bundle["exact_agreement"], expected)
        self.assertEqual(bundle["confusion"]["1"]["10"], sum(1 for t in y if t == 1))

    def test_always_10_macro_f1_zeros_missing_classes(self):
        _, rows = ev.load_dataset(root=ROOT)
        eligible = ev.eligible_records(rows)
        preds = ev.always_predict(10, eligible)
        bundle = ev.metrics_bundle(eligible, preds)
        y = [int(r["jev_choice"]) for r in eligible]
        rec = bundle["per_class_recall"]
        prec = bundle["per_class_precision"]
        f1s = []
        for k in ev.KEEP:
            p = prec[str(k)]
            r = rec[str(k)]
            if r != r:  # nan: no support
                f1s.append(0.0)
                continue
            if p != p:
                p = 0.0
            f1s.append(0.0 if (p + r) == 0 else 2 * p * r / (p + r))
        expected = sum(f1s) / 4
        self.assertAlmostEqual(bundle["macro_f1"], expected)
        self.assertEqual(prec["1"], 0.0)
        self.assertEqual(prec["3"], 0.0)
        self.assertEqual(prec["5"], 0.0)
        # Must not drop the three zero-F1 classes (old bug ≈ F1 of class 10 only).
        self.assertAlmostEqual(bundle["macro_f1"], f1s[3] / 4)
        self.assertLess(bundle["macro_f1"], bundle["exact_agreement"])
        self.assertEqual(len(set(y) & set(ev.KEEP)), 4)

    def test_layer_analysis_bands_and_always_correct(self):
        recs = []
        preds = []
        for layer, keep, pred in (
            (0, 10, 10),
            (0, 10, 10),
            (20, 5, 1),
            (20, 5, 1),
            (40, 10, 10),
            (40, 1, 10),
        ):
            recs.append(
                {
                    "sample_id": f"g:s2:l{layer}:{len(recs)}",
                    "generation_id": "g",
                    "step": 2,
                    "layer": layer,
                    "jev_choice": keep,
                    "layer_depth": layer / 49,
                    "previous_keep": 10,
                    "audio_residual_relative_l2": None,
                    "audio_rank": None,
                    "audio_cross_step_change": None,
                    "video_residual_relative_l2": None,
                    "video_rank": None,
                    "video_cross_step_change": None,
                    "jev_confidence": 0.9,
                    "jev_probabilities": None,
                    "applied_keep": keep,
                    "fallback_triggered": False,
                    "fallback_reason": None,
                    "quality_tag": "clean",
                    "prompt": "",
                    "num_steps": 4,
                }
            )
            preds.append(pred)
        la = ev.layer_analysis(recs, preds)
        self.assertIn(0, la["always_correct_layers"])
        self.assertIn(20, la["frequent_miss_layers"])
        self.assertEqual(la["per_layer"]["0"]["exact_agreement"], 1.0)
        self.assertEqual(la["per_layer"]["20"]["exact_agreement"], 0.0)
        self.assertEqual(la["bands"]["early_0_16"]["n"], 2)
        self.assertEqual(la["bands"]["mid_17_33"]["n"], 2)
        self.assertEqual(la["bands"]["late_34_49"]["n"], 2)
        self.assertEqual(la["per_layer"]["40"]["n_true1_pred10"], 1)


if __name__ == "__main__":
    unittest.main()
