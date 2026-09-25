"""CPU tests for Jev pair variability / hybrid budget. No GPU."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

RUNTIME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME))

import jev_variability as jv  # noqa: E402
import local_controller_eval as ev  # noqa: E402


def _rec(**kwargs):
    base = {
        "sample_id": "x",
        "generation_id": "g",
        "step": 2,
        "layer": 7,
        "jev_choice": 10,
        "layer_depth": 0.1,
        "previous_keep": 10,
        "audio_residual_relative_l2": 0.2,
        "audio_rank": 0.4,
        "audio_cross_step_change": None,
        "video_residual_relative_l2": 0.3,
        "video_rank": 0.5,
        "video_cross_step_change": None,
        "jev_confidence": 0.9,
        "jev_probabilities": None,
        "applied_keep": 10,
        "fallback_triggered": False,
        "fallback_reason": None,
        "quality_tag": "clean",
        "prompt": "",
        "num_steps": 4,
    }
    base.update(kwargs)
    base["sample_id"] = (
        f"{base['generation_id']}:s{base['step']}:l{base['layer']}:{base.get('_n', 0)}"
    )
    return base


class PairLabelTests(unittest.TestCase):
    def test_later_layer0_null_excluded_from_pair_n(self):
        rows = [
            _rec(
                generation_id="g1",
                step=2,
                layer=0,
                jev_choice=None,
                applied_keep=5,
                fallback_reason="block0_later_keep5",
                _n=0,
            ),
            _rec(generation_id="g1", step=2, layer=1, jev_choice=10, _n=1),
        ]
        table = jv.pair_cell_stats(rows)
        self.assertEqual(table["2:0"]["n"], 0)
        self.assertEqual(table["2:0"]["label"], "INSUFFICIENT")
        self.assertEqual(table["2:1"]["n"], 1)

    def test_always_10_cell_is_static(self):
        rows = [
            _rec(generation_id=f"g{i}", step=3, layer=42, jev_choice=10, _n=i)
            for i in range(12)
        ]
        table = jv.pair_cell_stats(rows)
        cell = table["3:42"]
        self.assertEqual(cell["n"], 12)
        self.assertEqual(cell["count_10"], 12)
        self.assertGreaterEqual(cell["majority_frac"], 0.95)
        self.assertEqual(cell["label"], "STATIC")

    def test_mixed_four_way_is_variable(self):
        keeps = [1, 3, 5, 10] * 3
        rows = [
            _rec(generation_id=f"g{i}", step=3, layer=22, jev_choice=keeps[i], _n=i)
            for i in range(12)
        ]
        table = jv.pair_cell_stats(rows)
        cell = table["3:22"]
        self.assertEqual(cell["n"], 12)
        self.assertEqual(cell["count_1"], 3)
        self.assertEqual(cell["count_3"], 3)
        self.assertEqual(cell["count_5"], 3)
        self.assertEqual(cell["count_10"], 3)
        self.assertLess(cell["majority_frac"], 0.55)
        self.assertEqual(cell["label"], "HIGHLY_VARIABLE")
        self.assertGreater(cell["entropy"], 1.5)


class LogoHoldoutTests(unittest.TestCase):
    def test_lookup_never_trains_on_held_out_generation(self):
        rows = [
            _rec(generation_id="gA", step=2, layer=7, jev_choice=1, _n=0),
            _rec(generation_id="gB", step=2, layer=7, jev_choice=10, _n=1),
        ]
        preds = jv.leave_one_gen_lookup(rows, jv.key_layer_step)
        self.assertEqual(preds[0], 10)
        self.assertEqual(preds[1], 1)
        self.assertEqual(preds, ev.layer_step_lookup_logo(rows))


class HybridBudgetTests(unittest.TestCase):
    def test_hybrid_parts_sum_to_eligible(self):
        rows = []
        n = 0
        for gid in ("gA", "gB", "gC", "gD", "gE", "gF"):
            for layer in range(50):
                rows.append(
                    _rec(
                        generation_id=gid,
                        step=1,
                        layer=layer,
                        jev_choice=10,
                        previous_keep=None,
                        _n=n,
                    )
                )
                n += 1
            for step in (2, 3, 4):
                for layer in range(50):
                    if layer == 0:
                        rows.append(
                            _rec(
                                generation_id=gid,
                                step=step,
                                layer=0,
                                jev_choice=None,
                                applied_keep=5,
                                fallback_reason="block0_later_keep5",
                                _n=n,
                            )
                        )
                    else:
                        keep = 10 if layer >= 40 else 5
                        rows.append(
                            _rec(
                                generation_id=gid,
                                step=step,
                                layer=layer,
                                jev_choice=keep,
                                _n=n,
                            )
                        )
                    n += 1
        eligible = ev.eligible_records(rows)
        table = jv.pair_cell_stats(rows)
        budget = jv.hybrid_budget(rows, table, fix_step1=True)
        self.assertEqual(budget["n_eligible"], len(eligible))
        self.assertEqual(
            budget["step1_fixed"]
            + budget["static_lookup_later"]
            + budget["needs_controller"],
            budget["n_eligible"],
        )
        self.assertEqual(budget["step1_fixed"], 6 * 50)
        self.assertGreater(budget["static_lookup_later"], 0)


def _noise_cell_rows(
    *,
    step: int,
    layer: int,
    specs: list[tuple[str, int, float, int | None, float, float]],
):
    """(generation_id, jev_choice, confidence, previous_keep, video_rank, residual)."""
    rows = []
    for i, (gid, keep, conf, prev, vrank, vres) in enumerate(specs):
        rows.append(
            _rec(
                generation_id=gid,
                step=step,
                layer=layer,
                jev_choice=keep,
                jev_confidence=conf,
                previous_keep=prev,
                video_rank=vrank,
                audio_rank=vrank,
                video_residual_relative_l2=vres,
                audio_residual_relative_l2=vres,
                _n=i,
            )
        )
    return rows


class VariableNoiseLabelTests(unittest.TestCase):
    def test_fixture_a_low_conf_mix_without_telemetry_is_likely_noise(self):
        # 12 gens, CONTENT-SENSITIVE: majority 10 at 7/12. Mix almost only at conf<0.5.
        specs = []
        low = [(1, 0.20), (5, 0.22), (10, 0.18), (5, 0.25), (10, 0.19), (1, 0.21), (5, 0.24), (10, 0.16)]
        for i, (keep, conf) in enumerate(low):
            specs.append((f"g{i}", keep, conf, 10, 0.5, 0.3))
        for i in range(8, 12):
            specs.append((f"g{i}", 10, 0.90, 10, 0.5, 0.3))
        rows = _noise_cell_rows(step=3, layer=11, specs=specs)
        table = jv.pair_cell_stats(rows)
        cell = table["3:11"]
        self.assertEqual(cell["n"], 12)
        self.assertEqual(cell["label"], "CONTENT_SENSITIVE")
        analysis = jv.analyze_variable_cells(rows, table)
        one = analysis["cells"]["3:11"]
        self.assertEqual(one["noise_label"], "LIKELY_JEV_NOISE")
        self.assertGreaterEqual(one["at_ge_0.5"]["majority_frac"], 0.80)
        self.assertLess(one["predictors"]["previous_keep_lift"], 0.10)
        self.assertLess(one["predictors"]["telem_all_lift"], 0.10)

    def test_fixture_b_high_conf_mix_predicted_by_previous_keep_is_real_signal(self):
        specs = []
        for i in range(6):
            specs.append((f"a{i}", 10, 0.88, 10, 0.75, 0.7))
        for i in range(6):
            specs.append((f"b{i}", 5, 0.86, 5, 0.15, 0.12))
        rows = _noise_cell_rows(step=4, layer=9, specs=specs)
        table = jv.pair_cell_stats(rows)
        cell = table["4:9"]
        self.assertEqual(cell["label"], "HIGHLY_VARIABLE")
        analysis = jv.analyze_variable_cells(rows, table)
        one = analysis["cells"]["4:9"]
        self.assertGreaterEqual(one["at_ge_0.8"]["n"], 5)
        self.assertLess(one["at_ge_0.8"]["majority_frac"], 0.80)
        self.assertGreaterEqual(one["predictors"]["previous_keep_lift"], 0.10)
        self.assertEqual(one["noise_label"], "REAL_DYNAMIC_SIGNAL")


class AggressiveBudgetTests(unittest.TestCase):
    def test_fixture_c_only_real_signal_stays_dynamic_and_parts_sum(self):
        rows = []
        n = 0
        gids = [f"g{i}" for i in range(6)]
        for gi, gid in enumerate(gids):
            for layer in range(50):
                rows.append(
                    _rec(
                        generation_id=gid,
                        step=1,
                        layer=layer,
                        jev_choice=10,
                        previous_keep=None,
                        jev_confidence=0.95,
                        _n=n,
                    )
                )
                n += 1
            for step in (2, 3, 4):
                for layer in range(50):
                    if layer == 0:
                        rows.append(
                            _rec(
                                generation_id=gid,
                                step=step,
                                layer=0,
                                jev_choice=None,
                                applied_keep=5,
                                fallback_reason="block0_later_keep5",
                                _n=n,
                            )
                        )
                    elif step == 3 and layer == 11:
                        # LIKELY NOISE: majority 10, two low-conf minorities.
                        if gi < 4:
                            keep, conf = 10, 0.91
                        else:
                            keep, conf = 1, 0.22
                        rows.append(
                            _rec(
                                generation_id=gid,
                                step=step,
                                layer=layer,
                                jev_choice=keep,
                                jev_confidence=conf,
                                previous_keep=10,
                                video_rank=0.5,
                                audio_rank=0.5,
                                video_residual_relative_l2=0.3,
                                audio_residual_relative_l2=0.3,
                                _n=n,
                            )
                        )
                    elif step == 3 and layer == 22:
                        # REAL SIGNAL: previous_keep splits 10 vs 5 at high conf.
                        if gi < 3:
                            keep, prev, vrank = 10, 10, 0.8
                        else:
                            keep, prev, vrank = 5, 5, 0.1
                        rows.append(
                            _rec(
                                generation_id=gid,
                                step=step,
                                layer=layer,
                                jev_choice=keep,
                                jev_confidence=0.87,
                                previous_keep=prev,
                                video_rank=vrank,
                                audio_rank=vrank,
                                video_residual_relative_l2=0.2 if keep == 5 else 0.7,
                                audio_residual_relative_l2=0.2 if keep == 5 else 0.7,
                                _n=n,
                            )
                        )
                    else:
                        keep = 10 if layer < 40 else 5
                        rows.append(
                            _rec(
                                generation_id=gid,
                                step=step,
                                layer=layer,
                                jev_choice=keep,
                                jev_confidence=0.92,
                                previous_keep=keep,
                                _n=n,
                            )
                        )
                    n += 1
        eligible = ev.eligible_records(rows)
        table = jv.pair_cell_stats(rows)
        self.assertEqual(table["3:11"]["label"], "CONTENT_SENSITIVE")
        self.assertEqual(table["3:22"]["label"], "HIGHLY_VARIABLE")
        analysis = jv.analyze_variable_cells(rows, table)
        self.assertEqual(analysis["cells"]["3:11"]["noise_label"], "LIKELY_JEV_NOISE")
        self.assertEqual(analysis["cells"]["3:22"]["noise_label"], "REAL_DYNAMIC_SIGNAL")
        budget = jv.aggressive_hybrid_budget(rows, table, analysis)
        self.assertEqual(budget["n_eligible"], len(eligible))
        self.assertEqual(
            budget["step1_fixed"]
            + budget["static_lookup_later"]
            + budget["noise_majority_lookup"]
            + budget["needs_controller"],
            budget["n_eligible"],
        )
        self.assertEqual(budget["step1_fixed"], 6 * 50)
        self.assertEqual(budget["needs_controller"], 6)
        self.assertEqual(budget["remaining_dynamic_cells"], ["3:22"])
        self.assertEqual(budget["remaining_dynamic_of_197"], 1)
        self.assertGreater(budget["pct_eliminated_of_197"], 99.0)
        self.assertEqual(budget["noise_majority_lookup"], 6)
        maj = jv.majority_keep_table(table)
        self.assertEqual(len(maj), 4)
        self.assertEqual(len(maj[0]), 50)
        self.assertTrue(all(k == 10 for k in maj[0]))
        self.assertEqual(maj[1][0], 5)
        self.assertEqual(maj[2][0], 5)
        self.assertEqual(maj[3][0], 5)
        self.assertEqual(maj[2][11], 10)
        self.assertEqual(maj[2][22], 10)


class HqTableFillTests(unittest.TestCase):
    """Shipped majority_keep_table conservative=True. Do not reimplement fill."""

    def test_n4_high_conf_majority_5_is_not_10(self):
        rows = [
            _rec(
                generation_id=f"g{i}",
                step=4,
                layer=12,
                jev_choice=5,
                jev_confidence=0.91,
                _n=i,
            )
            for i in range(4)
        ]
        table = jv.pair_cell_stats(rows)
        cell = table["4:12"]
        self.assertEqual(cell["n"], 4)
        self.assertEqual(cell["label"], "INSUFFICIENT")
        self.assertEqual(cell["majority"], 5)
        self.assertEqual(cell["high_conf_n"], 4)
        self.assertEqual(cell["high_conf_majority"], 5)
        reasons: list[list[str]] = []
        maj = jv.majority_keep_table(
            table,
            n_steps=4,
            force_step1_10=True,
            conservative=True,
            reasons=reasons,
        )
        self.assertEqual(maj[3][12], 5)
        self.assertNotEqual(maj[3][12], 10)
        self.assertEqual(reasons[3][12], "high_conf_majority")
        self.assertEqual(maj[1][0], 5)
        self.assertEqual(reasons[1][0], "later_layer0")
        self.assertTrue(all(k == 10 for k in maj[0]))
        self.assertEqual(reasons[0][12], "force_step1_10")

    def test_n4_high_conf_majority_3_is_not_10(self):
        rows = [
            _rec(
                generation_id=f"g{i}",
                step=3,
                layer=8,
                jev_choice=3,
                jev_confidence=0.88,
                _n=i,
            )
            for i in range(4)
        ]
        table = jv.pair_cell_stats(rows)
        self.assertEqual(table["3:8"]["label"], "INSUFFICIENT")
        maj = jv.majority_keep_table(
            table, n_steps=4, force_step1_10=True, conservative=True
        )
        self.assertEqual(maj[2][8], 3)
        self.assertNotEqual(maj[2][8], 10)
        self.assertEqual(maj[2][0], 5)

    def test_author_force_step1_10_still_10(self):
        rows = [
            _rec(
                generation_id=f"g{i}",
                step=1,
                layer=7,
                jev_choice=5,
                jev_confidence=0.95,
                previous_keep=None,
                _n=i,
            )
            for i in range(4)
        ]
        table = jv.pair_cell_stats(rows)
        forced = jv.majority_keep_table(
            table, n_steps=4, force_step1_10=True, conservative=True
        )
        observed = jv.majority_keep_table(
            table, n_steps=4, force_step1_10=False, conservative=True
        )
        self.assertEqual(forced[0][7], 10)
        self.assertEqual(observed[0][7], 5)


class HqTableArtifactTests(unittest.TestCase):
    def test_i2v_v2_table_mean_is_not_collapsed_to_10(self):
        root = Path(__file__).resolve().parents[3]
        path = root / "experiments" / "jev-hq-20step" / "tables" / "h3-i2v-20-sla-v2.json"
        self.assertTrue(path.is_file(), path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        table = payload["keep_table"]
        self.assertEqual(len(table), 20)
        self.assertEqual(len(table[0]), 50)
        vals = [int(k) for row in table for k in row]
        mean = sum(vals) / len(vals)
        self.assertLess(mean, 9.0)
        self.assertGreater(mean, 6.5)
        self.assertEqual(table[1][0], 5)
        self.assertTrue(any(int(k) in (1, 3, 5) for row in table[1:] for k in row[1:]))

    def test_pooled_family_table_mean_in_7_8_band(self):
        root = Path(__file__).resolve().parents[3]
        path = (
            root
            / "experiments"
            / "jev-hq-20step"
            / "tables"
            / "h3-fl2va-family-20-sla-v2.json"
        )
        self.assertTrue(path.is_file(), path)
        table = json.loads(path.read_text(encoding="utf-8"))["keep_table"]
        vals = [int(k) for row in table for k in row]
        mean = sum(vals) / len(vals)
        self.assertGreaterEqual(mean, 7.0)
        self.assertLessEqual(mean, 8.5)
        self.assertEqual(len(table), 20)
        self.assertTrue(all(int(k) == 10 for k in table[0]))
        self.assertTrue(all(int(row[0]) == 5 for row in table[1:]))


if __name__ == "__main__":
    unittest.main()
