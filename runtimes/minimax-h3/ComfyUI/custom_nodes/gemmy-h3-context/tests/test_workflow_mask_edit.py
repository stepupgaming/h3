import sys
import unittest
from pathlib import Path

COMFY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMFY))

from h3_workflow_build import (  # noqa: E402
    build_h3_workflow,
    build_sam3_track_workflow,
)


class WorkflowMaskEditTests(unittest.TestCase):
    def test_sam3_track_graph(self):
        graph = build_sam3_track_workflow(
            video_basename="src.mp4",
            mask_prompt="head",
            ckpt_name="sam3.1_multiplex_fp16.safetensors",
            length=124,
            object_indices="0",
            max_objects=1,
            filename_prefix="h3/gemmy_sam3",
        )
        by_type = {}
        for node in graph.values():
            by_type.setdefault(node["class_type"], []).append(node)
        self.assertEqual(len(by_type["SAM3_VideoTrack"]), 1)
        self.assertEqual(len(by_type["SAM3_TrackToMask"]), 1)
        self.assertEqual(by_type["SAM3_TrackToMask"][0]["inputs"]["object_indices"], "0")
        self.assertEqual(by_type["CLIPTextEncode"][0]["inputs"]["text"], "head")
        self.assertEqual(
            by_type["CheckpointLoaderSimple"][0]["inputs"]["ckpt_name"],
            "sam3.1_multiplex_fp16.safetensors",
        )
        self.assertIn("SaveVideo", by_type)

    def test_mask_edit_stage1_is_ganloss_ref_video(self):
        graph = build_h3_workflow(
            mode="ref2va",
            prompt="[video editing + reference generation]",
            width=608,
            height=352,
            length=39,
            steps=8,
            ref_images=["sheet.png"],
            ref_videos=["crop.mp4"],
            unet_name="10Eros_Max_h3_TURBO_ref2va_beta2_int8_convrot.safetensors",
            continuation_mode="ganloss_two_stage",
            persist_latent=True,
            cache="off",
        )
        by_type = {}
        for nid, node in graph.items():
            by_type.setdefault(node["class_type"], []).append((nid, node))
        self.assertNotIn("GemmyH3EncodeVideoFrames", by_type)
        self.assertNotIn("GemmyH3SetAVNoiseMask", by_type)
        self.assertNotIn("KSampler", by_type)
        ref = by_type["MiniMaxH3ReferenceToVideo"][0][1]["inputs"]
        self.assertEqual(ref["width"], 608)
        self.assertEqual(ref["height"], 352)
        self.assertIn("ref_images.ref_image_0", ref)
        self.assertIn("ref_videos.ref_video_0", ref)
        split = by_type["SplitSigmas"][0][1]["inputs"]
        self.assertEqual(split["step"], 4)
        samp = by_type["SamplerCustomAdvanced"][0][1]["inputs"]
        split_id = by_type["SplitSigmas"][0][0]
        self.assertEqual(samp["sigmas"], [split_id, 0])
        cond_id = by_type["MiniMaxH3ReferenceToVideo"][0][0]
        self.assertEqual(samp["latent_image"], [cond_id, 1])
        save = by_type["GemmyH3SaveAVLatent"][0][1]["inputs"]
        samp_id = by_type["SamplerCustomAdvanced"][0][0]
        self.assertEqual(save["samples"], [samp_id, 1])
        loads = {n["inputs"]["file"] for _, n in by_type["LoadVideo"]}
        self.assertIn("crop.mp4", loads)

    def test_mask_edit_stage2_is_3d_sr_plus_three_step(self):
        from h3_workflow_build import GANLOSS_REFINE_SIGMAS, GANLOSS_UPSCALE

        graph = build_h3_workflow(
            mode="ref2va",
            prompt="[video editing + reference generation]",
            width=1344,
            height=768,
            length=39,
            steps=3,
            ref_images=["sheet.png"],
            ref_videos=["crop.mp4"],
            continuation_mode="latent_refine",
            context_latent="stage1.h3av.safetensors",
            upscale_model_name=GANLOSS_UPSCALE,
            refine_sigmas=GANLOSS_REFINE_SIGMAS,
            persist_latent=False,
            cache="off",
        )
        by_type = {}
        for node in graph.values():
            by_type.setdefault(node["class_type"], []).append(node)
        self.assertEqual(len(by_type["MinimaxH3LatentUpscaler3D"]), 1)
        up = by_type["MinimaxH3LatentUpscaler3D"][0]["inputs"]
        self.assertEqual(up["mode.width"], 1344)
        self.assertEqual(up["mode.height"], 768)
        sig = by_type["ManualSigmas"][0]["inputs"]["sigmas"]
        self.assertEqual(sig, GANLOSS_REFINE_SIGMAS)
        self.assertEqual(by_type["KSamplerSelect"][0]["inputs"]["sampler_name"], "euler")
        self.assertIn("SamplerCustomAdvanced", by_type)


if __name__ == "__main__":
    unittest.main()
