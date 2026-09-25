/**
 * AUTHORING SOURCE. Do not hand-edit generated workflow.ir.json / comfy.workflow.json.
 *
 * Target environment: h3
 * Generated node SDK: environments/h3/nodes
 *
 * Rebuild: pnpm comfy:build
 */
import { workflow } from "@stepupgaming/comfy-workflows";
import { unsafeRef } from "@stepupgaming/comfy-workflows";
import type { Graph } from "@stepupgaming/comfy-workflows";
import {
  CreateVideo,
  GemmyH3JoinAV,
  GemmyH3LoadAVLatent,
  GemmyH3SaveAVLatent,
  GemmyH3SplitAV,
  MinimaxH3LatentUpscaler3D,
  SaveVideo,
  VAEDecode,
  VAEDecodeAudio,
  VAELoader,
} from "../environments/h3/nodes/registry.ts";

export function buildTemplate(): Graph {
  const g = workflow("comfy-workflow-h3-latent-preview");
  const context_latent = g.param("context_latent", { type: "string" });
  const upscale_model = g.param("upscale_model", { type: "string", default: "minimax_h3_latent_upscaler_3d_fp16.safetensors" });
  const scale = g.param("scale", { type: "int", default: 2 });
  const mode = g.param("mode", { type: "string", default: "scale by multiplier" });
  const target_width = g.param("target_width", { type: "int", default: 1280 });
  const target_height = g.param("target_height", { type: "int", default: 720 });
  const video_vae = g.param("video_vae", { type: "string", default: "minimax_h3_video_vae_fp16.safetensors" });
  const audio_vae = g.param("audio_vae", { type: "string", default: "minimax_h3_audio_vae_fp32.safetensors" });
  const output_prefix = g.param("output_prefix", { type: "string", default: "h3/latent_up" });
  const latent_prefix = g.param("latent_prefix", { type: "string", default: "h3/latent_up" });
  const fingerprint = g.param("fingerprint", { type: "string", default: "{}" });

  const gemmyH3LoadAVLatent_1 = g.add(GemmyH3LoadAVLatent, {
    "path": context_latent,
  }, { id: "1" });
  const gemmyH3SplitAV_2 = g.add(GemmyH3SplitAV, {
    "samples": unsafeRef(gemmyH3LoadAVLatent_1.id, 0),
  }, { id: "2" });
  const minimaxH3LatentUpscaler3D_3 = g.add(MinimaxH3LatentUpscaler3D, {
    "model_name": upscale_model,
    "mode": mode,
    "align": 32,
    "keep_proportion": true,
    "device": "cuda",
    "precision": "fp16",
    "latent": unsafeRef(gemmyH3SplitAV_2.id, 0),
  }, { id: "3" });
  g.setParamRaw(minimaxH3LatentUpscaler3D_3, "mode.scale", scale as never);
  g.setParamRaw(minimaxH3LatentUpscaler3D_3, "mode.width", target_width as never);
  g.setParamRaw(minimaxH3LatentUpscaler3D_3, "mode.height", target_height as never);
  const gemmyH3JoinAV_4 = g.add(GemmyH3JoinAV, {
    "video": unsafeRef(minimaxH3LatentUpscaler3D_3.id, 0),
    "audio": unsafeRef(gemmyH3SplitAV_2.id, 1),
  }, { id: "4" });
  const vaeLoader_5 = g.add(VAELoader, {
    "vae_name": video_vae,
  }, { id: "5" });
  const vaeLoader_6 = g.add(VAELoader, {
    "vae_name": audio_vae,
  }, { id: "6" });
  const vaeDecode_7 = g.add(VAEDecode, {
    "samples": unsafeRef(gemmyH3JoinAV_4.id, 0),
    "vae": unsafeRef(vaeLoader_5.id, 0),
  }, { id: "7" });
  const vaeDecodeAudio_8 = g.add(VAEDecodeAudio, {
    "samples": unsafeRef(gemmyH3JoinAV_4.id, 0),
    "vae": unsafeRef(vaeLoader_6.id, 0),
  }, { id: "8" });
  const createVideo_9 = g.add(CreateVideo, {
    "fps": 24,
    "images": unsafeRef(vaeDecode_7.id, 0),
    "audio": unsafeRef(vaeDecodeAudio_8.id, 0),
  }, { id: "9" });
  const saveVideo_10 = g.add(SaveVideo, {
    "filename_prefix": output_prefix,
    "format": "auto",
    "codec": "auto" as never,
    "video": unsafeRef(createVideo_9.id, 0),
  }, { id: "10" });
  const gemmyH3SaveAVLatent_11 = g.add(GemmyH3SaveAVLatent, {
    "filename_prefix": latent_prefix,
    "fingerprint": fingerprint,
    "samples": unsafeRef(gemmyH3JoinAV_4.id, 0),
  }, { id: "11" });
  g.output(saveVideo_10.out(0));
  g.output(gemmyH3SaveAVLatent_11.out(0));
  return g.toGraph();
}

export default buildTemplate;

export const manifestMeta = {
  name: "comfy-workflow-h3-latent-preview",
  title: "H3 Latent Upscale Preview",
  description: "3D latent upscale then decode only (no second H3 sample).",
  outputs: [
  {
    "name": "output-0",
    "type": "IMAGE"
  },
  {
    "name": "output-1",
    "type": "IMAGE"
  }
],
  models: [
  {
    "kind": "model",
    "name": "minimax_h3_latent_upscaler_3d_fp16.safetensors"
  },
  {
    "kind": "model",
    "name": "minimax_h3_video_vae_fp16.safetensors"
  },
  {
    "kind": "model",
    "name": "minimax_h3_audio_vae_fp32.safetensors"
  }
],
  environment: "h3",
  nodePacks: ["gemmy-h3-context","Comfyui_Minimax_h3_latent_Upscaler"],
};
