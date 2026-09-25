import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from h3_mask_edit import (  # noqa: E402
    apply_crop,
    cleanup_masks,
    compose_overlay,
    mask_coverage,
    size_for_megapixels,
    pixel_mask_to_h3_latent_mask,
    plan_combined_crop,
    plan_tracked_crop,
    uncrop,
    video_latent_t,
)


class MaskEditMathTests(unittest.TestCase):
    def test_combined_crop_covers_travel(self):
        masks = np.zeros((8, 64, 96), dtype=np.float32)
        masks[0, 8:16, 8:16] = 1
        masks[7, 40:56, 64:88] = 1
        plan = plan_combined_crop(masks, crop_scale=1.0, divisible_by=8, upscale_megapixels=0)
        box = plan["boxes"][0]
        self.assertEqual(box, plan["boxes"][-1])
        self.assertLessEqual(box[0], 8)
        self.assertLessEqual(box[1], 8)
        self.assertGreaterEqual(box[2], 88)
        self.assertGreaterEqual(box[3], 56)
        self.assertEqual(plan["out_width"] % 8, 0)
        self.assertEqual(plan["out_height"] % 8, 0)

    def test_empty_mask_fails(self):
        masks = np.zeros((4, 32, 32), dtype=np.float32)
        with self.assertRaises(ValueError):
            plan_combined_crop(masks)

    def test_tracked_moves_only_when_needed(self):
        masks = np.zeros((6, 64, 64), dtype=np.float32)
        masks[0:3, 8:16, 8:16] = 1
        masks[3:6, 40:52, 40:52] = 1
        plan = plan_tracked_crop(masks, crop_scale=1.2, divisible_by=8)
        first = tuple(plan["boxes"][0])
        self.assertEqual(tuple(plan["boxes"][1]), first)
        self.assertNotEqual(tuple(plan["boxes"][-1]), first)
        self.assertEqual(plan["out_width"], plan["boxes"][0][2] - plan["boxes"][0][0])

    def test_crop_uncrop_roundtrip_identity_inside_box(self):
        rng = np.random.default_rng(0)
        frames = rng.random((5, 48, 64, 3), dtype=np.float32)
        masks = np.zeros((5, 48, 64), dtype=np.float32)
        masks[:, 10:30, 12:44] = 1
        plan = plan_combined_crop(masks, crop_scale=1.0, divisible_by=8)
        cropped_f, cropped_m = apply_crop(frames, masks, plan)
        self.assertEqual(cropped_f.shape[1], plan["out_height"])
        self.assertEqual(cropped_f.shape[2], plan["out_width"])
        rebuilt = uncrop(cropped_f, frames, plan, cropped_m, feather=0)
        x0, y0, x1, y1 = plan["boxes"][0]
        np.testing.assert_allclose(
            rebuilt[:, y0:y1, x0:x1], frames[:, y0:y1, x0:x1], atol=1e-5
        )

    def test_latent_mask_token_grid(self):
        # 39 frames → 12 latent T; 64×64 → 4×4 tokens at /16.
        masks = np.zeros((39, 64, 64), dtype=np.float32)
        masks[:, 0:16, 0:16] = 1
        z = pixel_mask_to_h3_latent_mask(masks)
        self.assertEqual(z.shape, (1, 1, video_latent_t(39), 4, 4))
        self.assertGreater(float(z[0, 0, 0, 0, 0]), 0.5)
        self.assertEqual(float(z[0, 0, 0, 3, 3]), 0.0)

    def test_cleanup_drops_single_frame_flash(self):
        masks = np.zeros((6, 32, 32), dtype=np.float32)
        masks[:, 8:24, 8:24] = 1
        masks[2, 0:2, 0:2] = 1
        cleaned = cleanup_masks(masks, min_area=4, min_frame_presence=2)
        self.assertLess(float(cleaned[2, 0, 0]), 0.5)

    def test_overlay_tints_only_the_mask(self):
        frames = np.full((2, 8, 8, 3), 0.4, dtype=np.float32)
        masks = np.zeros((2, 8, 8), dtype=np.float32)
        masks[:, 0:4, 0:4] = 1
        overlay = compose_overlay(frames, masks, tint=(1.0, 0.0, 0.0), alpha=0.5)
        np.testing.assert_allclose(overlay[0, 6, 6], [0.4, 0.4, 0.4], atol=1e-5)
        np.testing.assert_allclose(overlay[1, 6, 6], [0.4, 0.4, 0.4], atol=1e-5)
        self.assertGreater(float(overlay[0, 1, 1, 0]), 0.6)
        self.assertLess(float(overlay[0, 1, 1, 1]), 0.4)
        cov = mask_coverage(masks)
        self.assertEqual(cov["frames"], 2)
        self.assertAlmostEqual(float(cov["mean"]), 0.25, places=5)

    def test_size_for_megapixels_can_shrink(self):
        w, h = size_for_megapixels(1312, 768, 0.2, 32)
        self.assertEqual(w % 32, 0)
        self.assertEqual(h % 32, 0)
        self.assertLess(w * h, 1312 * 768)
        self.assertAlmostEqual(w * h / 1_000_000, 0.2, delta=0.05)

    def test_uncrop_feather_keeps_subject_interior(self):
        # Plate: dark square (source hair) on green. Generated: smaller orange
        # subject on the same green — no dark pixels. A mask that covers the
        # dark square must paste generated green in the leftover hair ring,
        # not blend the dark plate back in.
        h, w = 64, 64
        originals = np.zeros((1, h, w, 3), dtype=np.float32)
        originals[..., 1] = 0.6
        originals[0, 10:50, 10:50] = 0.02
        masks = np.zeros((1, h, w), dtype=np.float32)
        masks[0, 10:50, 10:50] = 1.0
        processed = np.zeros((1, h, w, 3), dtype=np.float32)
        processed[..., 1] = 0.6
        processed[0, 18:42, 18:42] = np.array([0.95, 0.35, 0.15], dtype=np.float32)
        plan = {
            "boxes": [[0, 0, w, h]],
            "out_width": w,
            "out_height": h,
            "source_width": w,
            "source_height": h,
        }
        out = uncrop(processed, originals, plan, masks, feather=8)
        np.testing.assert_allclose(out[0, 30, 30], [0.95, 0.35, 0.15], atol=0.05)
        ring = out[0, 12, 30]
        self.assertGreater(float(ring[1]), 0.45)
        self.assertLess(float(ring[0]), 0.2)
        np.testing.assert_allclose(out[0, 2, 2], [0.0, 0.6, 0.0], atol=0.05)

    def test_uncrop_does_not_print_old_silhouette(self):
        # Old dark hair volume is larger than the new subject. Matching
        # hedges outside the SAM3 matte must come from the crop so the old
        # outline does not sit in the background. A changed unmasked person
        # must stay original.
        h, w = 64, 80
        originals = np.zeros((1, h, w, 3), dtype=np.float32)
        originals[..., 1] = 0.5
        originals[0, 8:48, 8:40] = 0.02
        originals[0, 8:48, 56:76] = np.array([0.9, 0.9, 0.85], dtype=np.float32)
        masks = np.zeros((1, h, w), dtype=np.float32)
        masks[0, 8:48, 8:40] = 1.0
        processed = np.zeros((1, h, w, 3), dtype=np.float32)
        processed[..., 1] = 0.5
        processed[0, 16:40, 14:34] = np.array([0.95, 0.35, 0.15], dtype=np.float32)
        processed[0, 8:48, 56:76] = np.array([0.1, 0.2, 0.8], dtype=np.float32)
        plan = {
            "boxes": [[0, 0, w, h]],
            "out_width": w,
            "out_height": h,
            "source_width": w,
            "source_height": h,
        }
        out = uncrop(processed, originals, plan, masks, feather=8)
        np.testing.assert_allclose(out[0, 28, 24], [0.95, 0.35, 0.15], atol=0.05)
        self.assertGreater(float(out[0, 10, 10, 1]), 0.35)
        self.assertLess(float(out[0, 10, 10, 0]), 0.2)
        self.assertGreater(float(out[0, 28, 4, 1]), 0.35)
        np.testing.assert_allclose(out[0, 28, 66], [0.9, 0.9, 0.85], atol=0.08)


if __name__ == "__main__":
    unittest.main()
