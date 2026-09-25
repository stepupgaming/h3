import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from av_math import (
    MAX_FRACTIONAL_CHANGE,
    audio_latent_t,
    audio_tail_slice,
    conform_audio_ok,
    expected_audio_samples,
    fingerprint,
    fingerprints_match,
    fit_audio_time,
    native_guide_tail_spec,
    snap_context_frames,
    video_latent_t,
)


class AvMathTests(unittest.TestCase):
    def test_default_context_is_39(self):
        self.assertEqual(snap_context_frames(39, 124), 39)
        self.assertEqual(video_latent_t(39), 12)
        self.assertEqual(audio_latent_t(39), 65)

    def test_native_guide_tail_spec_fits_new_clip(self):
        ctx, vt, at = native_guide_tail_spec(39, 124, 124)
        self.assertEqual((ctx, vt, at), (39, 12, 65))
        ctx, vt, at = native_guide_tail_spec(141, 90, 124)
        self.assertEqual(ctx, 90)
        self.assertEqual(vt, video_latent_t(90))

    def test_snap_down_to_shared_boundary(self):
        self.assertEqual(snap_context_frames(100, 200), 90)
        self.assertEqual(snap_context_frames(141, 141), 141)
        self.assertEqual(snap_context_frames(50, 40), 39)

    def test_fingerprint_mismatch_on_canvas(self):
        a = fingerprint(width=864, height=480, dit="fl2va", mode="i2v")
        b = fingerprint(width=1280, height=736, dit="fl2va", mode="i2v")
        self.assertFalse(fingerprints_match(a, b))
        self.assertTrue(fingerprints_match(a, fingerprint(width=864, height=480, dit="fl2va", mode="i2v")))

    def test_fingerprint_mismatch_on_vae(self):
        a = fingerprint(width=864, height=480, dit="fl2va", mode="i2v")
        b = fingerprint(
            width=864,
            height=480,
            dit="fl2va",
            mode="i2v",
            vae="minimax_h3_video_vae_fp16",
        )
        self.assertFalse(fingerprints_match(a, b))
        self.assertEqual(a["vae"], "minimax_h3_video_vae_int8_convrot")

    def test_audio_conform_bound(self):
        expected = 78000
        self.assertTrue(conform_audio_ok(expected, expected))
        slop = int(expected * MAX_FRACTIONAL_CHANGE)
        self.assertTrue(conform_audio_ok(expected + slop, expected))
        self.assertFalse(conform_audio_ok(expected + slop + 20, expected))

    def test_audio_tail_slice_imported_mp4(self):
        # 124-frame import @ 32 kHz vs 39-frame VAE-tail (the live fail).
        expected = expected_audio_samples(39, 32000)
        self.assertEqual(expected, 52000)
        start, end = audio_tail_slice(165888, 39, 32000)
        self.assertEqual((start, end), (165888 - 52000, 165888))
        short_start, short_end = audio_tail_slice(40000, 39, 32000)
        self.assertEqual((short_start, short_end), (0, 40000))

    def test_fit_audio_time_pad_and_crop(self):
        self.assertEqual(fit_audio_time(134, 150), (134, 16))
        self.assertEqual(fit_audio_time(150, 150), (150, 0))
        self.assertEqual(fit_audio_time(180, 150), (150, 0))
        with self.assertRaises(ValueError):
            fit_audio_time(0, 150)


if __name__ == "__main__":
    unittest.main()
