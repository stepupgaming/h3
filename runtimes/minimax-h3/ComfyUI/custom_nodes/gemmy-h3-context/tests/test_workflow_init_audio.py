import sys
import unittest
from pathlib import Path

COMFY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMFY))

from h3_workflow_build import build_h3_workflow  # noqa: E402


class WorkflowInitAudioTests(unittest.TestCase):
    def test_ref2va_init_audio_img2img(self):
        graph = build_h3_workflow(
            mode="ref2va",
            prompt="test",
            width=64,
            height=64,
            length=90,
            ref_images=["dark.png"],
            ref_audios=["female.wav"],
            init_audio="male.wav",
            denoise=0.55,
            persist_latent=False,
        )
        by_type = {}
        for node in graph.values():
            by_type.setdefault(node["class_type"], []).append(node)
        self.assertEqual(len(by_type.get("GemmyH3InitFromAudio", [])), 1)
        ks = by_type.get("KSampler") or []
        self.assertEqual(len(ks), 1)
        self.assertEqual(ks[0]["inputs"]["denoise"], 0.55)
        init_id = next(
            nid for nid, node in graph.items() if node["class_type"] == "GemmyH3InitFromAudio"
        )
        self.assertEqual(ks[0]["inputs"]["latent_image"], [init_id, 0])
        load_audio = by_type.get("LoadAudio") or []
        names = {n["inputs"]["audio"] for n in load_audio}
        self.assertIn("male.wav", names)
        self.assertIn("female.wav", names)


    def test_native_guide_uses_tail_guide_not_mask(self):
        graph = build_h3_workflow(
            mode="t2va",
            prompt="test",
            width=864,
            height=480,
            length=124,
            continuation_mode="native_guide",
            context_latent=r"C:\tmp\clip.h3av.safetensors",
            context_frames=39,
            persist_latent=False,
        )
        types = {node["class_type"] for node in graph.values()}
        self.assertIn("GemmyH3LatentTailGuide", types)
        self.assertIn("GemmyH3LoadAVLatent", types)
        self.assertNotIn("GemmyH3MaskedAVContext", types)
        self.assertNotIn("GemmyH3TrimProtectedPrefix", types)
        ks = [n for n in graph.values() if n["class_type"] == "KSampler"]
        self.assertEqual(len(ks), 1)
        guide_id = next(
            nid for nid, node in graph.items() if node["class_type"] == "GemmyH3LatentTailGuide"
        )
        self.assertEqual(ks[0]["inputs"]["positive"], [guide_id, 0])


if __name__ == "__main__":
    unittest.main()
