import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stitch_continuation import find_cut_frame, stitch_tensors  # noqa: E402


class StitchContinuationTests(unittest.TestCase):
    def test_find_cut_frame_on_copied_tail(self):
        torch.manual_seed(0)
        source = torch.rand(50, 24, 32, 3)
        start = 11
        follow = torch.cat([source[start:start + 20].clone(), torch.rand(10, 24, 32, 3)], dim=0)
        cut, score = find_cut_frame(source, follow, search_frames=40, match_window=8)
        self.assertEqual(cut, start)
        self.assertGreater(score, 0.99)

    def test_stitch_drops_source_overlap_and_keeps_continuation(self):
        torch.manual_seed(1)
        source = torch.rand(124, 16, 16, 3)
        overlap = 39
        follow = torch.cat(
            [source[-overlap:].clone(), torch.rand(124 - overlap, 16, 16, 3)],
            dim=0,
        )
        out = stitch_tensors(
            source, follow,
            overlap_frames=overlap,
            alignment="fixed overlap",
            color_match="off",
            window_audio="from continuation",
        )
        self.assertEqual(out["cut_frame"], 124 - overlap)
        self.assertEqual(int(out["frames"].shape[0]), (124 - overlap) + 124)

    def test_auto_alignment_finds_nominal_overlap(self):
        torch.manual_seed(2)
        source = torch.rand(80, 16, 16, 3)
        overlap = 20
        follow = torch.cat(
            [source[-overlap:].clone() * 0.98 + 0.01, torch.rand(40, 16, 16, 3)],
            dim=0,
        )
        out = stitch_tensors(
            source, follow,
            overlap_frames=overlap,
            alignment="auto",
            search_frames=10,
            match_window=8,
            color_match="off",
        )
        self.assertAlmostEqual(out["cut_frame"], 80 - overlap, delta=2)
        self.assertGreater(out["match"], 0.9)


if __name__ == "__main__":
    unittest.main()
