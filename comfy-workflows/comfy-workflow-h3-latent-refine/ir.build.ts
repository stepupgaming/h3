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
  BasicGuider,
  CLIPLoader,
  ConditioningZeroOut,
  CreateVideo,
  GemmyH3JoinAV,
  GemmyH3LoadAVLatent,
  GemmyH3SaveAVLatent,
  GemmyH3SplitAV,
  KSamplerSelect,
  ManualSigmas,
  MiniMaxH3ImageToVideo,
  MiniMaxH3SigmaShift,
  MinimaxH3LatentUpscaler3D,
  RandomNoise,
  SamplerCustomAdvanced,
  SaveVideo,
  UNETLoader,
  VAEDecode,
  VAEDecodeAudio,
  VAELoader,
} from "../environments/h3/nodes/registry.ts";

export function buildTemplate(): Graph {
  const g = workflow("comfy-workflow-h3-latent-refine");
  const unet = g.param("unet", { type: "string", default: "minimax_h3_fl2va_pruned_int8_convrot.safetensors" });
  const clip = g.param("clip", { type: "string", default: "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" });
  const video_vae = g.param("video_vae", { type: "string", default: "minimax_h3_video_vae_fp16.safetensors" });
  const audio_vae = g.param("audio_vae", { type: "string", default: "minimax_h3_audio_vae_fp32.safetensors" });
  const prompt = g.param("prompt", { type: "string" });
  const width = g.param("width", { type: "int", default: 1280 });
  const height = g.param("height", { type: "int", default: 720 });
  const length = g.param("length", { type: "int", default: 124 });
  const context_latent = g.param("context_latent", { type: "string" });
  const upscale_model = g.param("upscale_model", { type: "string", default: "minimax_h3_latent_upscaler_3d_fp16.safetensors" });
  const seed = g.param("seed", { type: "int", default: 42 });
  const shift_video = g.param("shift_video", { type: "int", default: 12 });
  const shift_audio = g.param("shift_audio", { type: "int", default: 3 });
  const latent_prefix = g.param("latent_prefix", { type: "string", default: "h3/latent_refine_av" });
  const output_prefix = g.param("output_prefix", { type: "string", default: "h3/latent_refine" });
  const fingerprint = g.param("fingerprint", { type: "string", default: "{}" });

  const unetLoader_1 = g.add(UNETLoader, {
    "unet_name": unet,
    "weight_dtype": "default",
  }, { id: "1" });
  const clipLoader_2 = g.add(CLIPLoader, {
    "clip_name": clip,
    "type": "minimax",
    "device": "cpu",
  }, { id: "2" });
  const vaeLoader_3 = g.add(VAELoader, {
    "vae_name": video_vae,
  }, { id: "3" });
  const vaeLoader_4 = g.add(VAELoader, {
    "vae_name": audio_vae,
  }, { id: "4" });
  const miniMaxH3SigmaShift_5 = g.add(MiniMaxH3SigmaShift, {
    "shift_video": shift_video,
    "shift_audio": shift_audio,
    "model": unsafeRef(unetLoader_1.id, 0),
  }, { id: "5" });
  const miniMaxH3ImageToVideo_6 = g.add(MiniMaxH3ImageToVideo, {
    "prompt": prompt,
    "width": width,
    "height": height,
    "length": length,
    "clip": unsafeRef(clipLoader_2.id, 0),
    "vae": unsafeRef(vaeLoader_3.id, 0),
  }, { id: "6" });
  const gemmyH3LoadAVLatent_7 = g.add(GemmyH3LoadAVLatent, {
    "path": context_latent,
  }, { id: "7" });
  const gemmyH3SplitAV_8 = g.add(GemmyH3SplitAV, {
    "samples": unsafeRef(gemmyH3LoadAVLatent_7.id, 0),
  }, { id: "8" });
  const minimaxH3LatentUpscaler3D_9 = g.add(MinimaxH3LatentUpscaler3D, {
    "model_name": upscale_model,
    "mode": "target dimensions" as never,
    "align": 32,
    "keep_proportion": true,
    "device": "cuda",
    "precision": "fp16",
    "latent": unsafeRef(gemmyH3SplitAV_8.id, 0),
  }, { id: "9" });
  g.setParamRaw(minimaxH3LatentUpscaler3D_9, "mode.width", 1280 as never);
  g.setParamRaw(minimaxH3LatentUpscaler3D_9, "mode.height", 720 as never);
  const gemmyH3JoinAV_10 = g.add(GemmyH3JoinAV, {
    "video": unsafeRef(minimaxH3LatentUpscaler3D_9.id, 0),
    "audio": unsafeRef(gemmyH3SplitAV_8.id, 1),
  }, { id: "10" });
  const conditioningZeroOut_11 = g.add(ConditioningZeroOut, {
    "conditioning": unsafeRef(miniMaxH3ImageToVideo_6.id, 0),
  }, { id: "11" });
  const randomNoise_12 = g.add(RandomNoise, {
    "noise_seed": seed,
  }, { id: "12" });
  const basicGuider_13 = g.add(BasicGuider, {
    "model": unsafeRef(miniMaxH3SigmaShift_5.id, 0),
    "conditioning": unsafeRef(miniMaxH3ImageToVideo_6.id, 0),
  }, { id: "13" });
  const manualSigmas_14 = g.add(ManualSigmas, {
    "sigmas": "0.9035, 0.6316, 0.3158, 0.0000",
  }, { id: "14" });
  const kSamplerSelect_15 = g.add(KSamplerSelect, {
    "sampler_name": "euler",
  }, { id: "15" });
  const samplerCustomAdvanced_16 = g.add(SamplerCustomAdvanced, {
    "noise": unsafeRef(randomNoise_12.id, 0),
    "guider": unsafeRef(basicGuider_13.id, 0),
    "sampler": unsafeRef(kSamplerSelect_15.id, 0),
    "sigmas": unsafeRef(manualSigmas_14.id, 0),
    "latent_image": unsafeRef(gemmyH3JoinAV_10.id, 0),
  }, { id: "16" });
  const gemmyH3SaveAVLatent_17 = g.add(GemmyH3SaveAVLatent, {
    "filename_prefix": latent_prefix,
    "fingerprint": fingerprint,
    "samples": unsafeRef(samplerCustomAdvanced_16.id, 0),
  }, { id: "17" });
  const vaeDecode_18 = g.add(VAEDecode, {
    "samples": unsafeRef(samplerCustomAdvanced_16.id, 0),
    "vae": unsafeRef(vaeLoader_3.id, 0),
  }, { id: "18" });
  const vaeDecodeAudio_19 = g.add(VAEDecodeAudio, {
    "samples": unsafeRef(samplerCustomAdvanced_16.id, 0),
    "vae": unsafeRef(vaeLoader_4.id, 0),
  }, { id: "19" });
  const createVideo_20 = g.add(CreateVideo, {
    "fps": 24,
    "images": unsafeRef(vaeDecode_18.id, 0),
    "audio": unsafeRef(vaeDecodeAudio_19.id, 0),
  }, { id: "20" });
  const saveVideo_21 = g.add(SaveVideo, {
    "filename_prefix": output_prefix,
    "format": "auto",
    "codec": "auto" as never,
    "video": unsafeRef(createVideo_20.id, 0),
  }, { id: "21" });
  g.output(conditioningZeroOut_11.out(0));
  g.output(gemmyH3SaveAVLatent_17.out(0));
  g.output(saveVideo_21.out(0));
  return g.toGraph();
}

export default buildTemplate;

export const manifestMeta = {
  name: "comfy-workflow-h3-latent-refine",
  title: "H3 Latent Upscale + Refine",
  description: "3D latent upscale then a short H3 refine sample. Gemmy `video h3 upscale --backend latent`.",
  outputs: [
  {
    "name": "output-0",
    "type": "IMAGE"
  },
  {
    "name": "output-1",
    "type": "IMAGE"
  },
  {
    "name": "output-2",
    "type": "IMAGE"
  }
],
  models: [
  {
    "kind": "model",
    "name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
  },
  {
    "kind": "model",
    "name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
  },
  {
    "kind": "model",
    "name": "minimax_h3_video_vae_fp16.safetensors"
  },
  {
    "kind": "model",
    "name": "minimax_h3_audio_vae_fp32.safetensors"
  },
  {
    "kind": "model",
    "name": "minimax_h3_latent_upscaler_3d_fp16.safetensors"
  }
],
  environment: "h3",
  nodePacks: ["gemmy-h3-context","Comfyui_Minimax_h3_latent_Upscaler"],
};
