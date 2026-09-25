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
  CLIPLoader,
  ConditioningZeroOut,
  CreateVideo,
  GemmyH3LoadAVLatent,
  GemmyH3MaskedAVContext,
  GemmyH3SaveAVLatent,
  GemmyH3TrimProtectedPrefix,
  KSampler,
  LoadImage,
  MiniMaxH3ImageToVideo,
  MiniMaxH3SigmaShift,
  SaveVideo,
  UNETLoader,
  VAEDecode,
  VAEDecodeAudio,
  VAELoader,
} from "../environments/h3/nodes/registry.ts";

export function buildTemplate(): Graph {
  const g = workflow("comfy-workflow-h3-continue");
  const unet = g.param("unet", { type: "string", default: "minimax_h3_fl2va_pruned_int8_convrot.safetensors" });
  const clip = g.param("clip", { type: "string", default: "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" });
  const video_vae = g.param("video_vae", { type: "string", default: "minimax_h3_video_vae_fp16.safetensors" });
  const audio_vae = g.param("audio_vae", { type: "string", default: "minimax_h3_audio_vae_fp32.safetensors" });
  const first_image = g.param("first_image", { type: "combo" });
  const context_latent = g.param("context_latent", { type: "string" });
  const context_frames = g.param("context_frames", { type: "int", default: 39 });
  const prompt = g.param("prompt", { type: "string" });
  const width = g.param("width", { type: "int", default: 864 });
  const height = g.param("height", { type: "int", default: 480 });
  const length = g.param("length", { type: "int", default: 124 });
  const seed = g.param("seed", { type: "int", default: 42 });
  const steps = g.param("steps", { type: "int", default: 20 });
  const denoise = g.param("denoise", { type: "float", default: 1 });
  const shift_video = g.param("shift_video", { type: "int", default: 12 });
  const shift_audio = g.param("shift_audio", { type: "int", default: 3 });
  const latent_prefix = g.param("latent_prefix", { type: "string", default: "h3/continue_av" });
  const output_prefix = g.param("output_prefix", { type: "string", default: "h3/continue" });
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
  const loadImage_6 = g.add(LoadImage, {
    "image": first_image,
  }, { id: "6" });
  const miniMaxH3ImageToVideo_7 = g.add(MiniMaxH3ImageToVideo, {
    "prompt": prompt,
    "width": width,
    "height": height,
    "length": length,
    "clip": unsafeRef(clipLoader_2.id, 0),
    "vae": unsafeRef(vaeLoader_3.id, 0),
    "first_frame": unsafeRef(loadImage_6.id, 0),
  }, { id: "7" });
  const gemmyH3LoadAVLatent_8 = g.add(GemmyH3LoadAVLatent, {
    "path": context_latent,
  }, { id: "8" });
  const gemmyH3MaskedAVContext_9 = g.add(GemmyH3MaskedAVContext, {
    "context_frames": context_frames,
    "audio_mode": "generated_audio",
    "target": unsafeRef(miniMaxH3ImageToVideo_7.id, 1),
    "vae": unsafeRef(vaeLoader_3.id, 0),
    "audio_vae": unsafeRef(vaeLoader_4.id, 0),
    "previous": unsafeRef(gemmyH3LoadAVLatent_8.id, 0),
  }, { id: "9" });
  const conditioningZeroOut_10 = g.add(ConditioningZeroOut, {
    "conditioning": unsafeRef(miniMaxH3ImageToVideo_7.id, 0),
  }, { id: "10" });
  const kSampler_11 = g.add(KSampler, {
    "seed": seed,
    "steps": steps,
    "cfg": 1,
    "sampler_name": "euler",
    "scheduler": "simple",
    "denoise": denoise,
    "model": unsafeRef(miniMaxH3SigmaShift_5.id, 0),
    "positive": unsafeRef(miniMaxH3ImageToVideo_7.id, 0),
    "negative": unsafeRef(conditioningZeroOut_10.id, 0),
    "latent_image": unsafeRef(gemmyH3MaskedAVContext_9.id, 0),
  }, { id: "11" });
  const gemmyH3SaveAVLatent_12 = g.add(GemmyH3SaveAVLatent, {
    "filename_prefix": latent_prefix,
    "fingerprint": fingerprint,
    "samples": unsafeRef(kSampler_11.id, 0),
  }, { id: "12" });
  const vaeDecode_13 = g.add(VAEDecode, {
    "samples": unsafeRef(kSampler_11.id, 0),
    "vae": unsafeRef(vaeLoader_3.id, 0),
  }, { id: "13" });
  const vaeDecodeAudio_14 = g.add(VAEDecodeAudio, {
    "samples": unsafeRef(kSampler_11.id, 0),
    "vae": unsafeRef(vaeLoader_4.id, 0),
  }, { id: "14" });
  const gemmyH3TrimProtectedPrefix_15 = g.add(GemmyH3TrimProtectedPrefix, {
    "images": unsafeRef(vaeDecode_13.id, 0),
    "protected_frames": unsafeRef(gemmyH3MaskedAVContext_9.id, 1) as never,
    "audio": unsafeRef(vaeDecodeAudio_14.id, 0),
  }, { id: "15" });
  const createVideo_16 = g.add(CreateVideo, {
    "fps": 24,
    "images": unsafeRef(gemmyH3TrimProtectedPrefix_15.id, 0),
    "audio": unsafeRef(gemmyH3TrimProtectedPrefix_15.id, 1),
  }, { id: "16" });
  const saveVideo_17 = g.add(SaveVideo, {
    "filename_prefix": output_prefix,
    "format": "auto",
    "codec": "auto" as never,
    "video": unsafeRef(createVideo_16.id, 0),
  }, { id: "17" });
  g.output(gemmyH3SaveAVLatent_12.out(0));
  g.output(saveVideo_17.out(0));
  return g.toGraph();
}

export default buildTemplate;

export const manifestMeta = {
  name: "comfy-workflow-h3-continue",
  title: "H3 Masked-AV Continue (FL2VA)",
  description: "Continue an H3 clip from a persisted AV latent plus first frame. Gemmy continue/loop.",
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
  }
],
  environment: "h3",
  nodePacks: ["gemmy-h3-context"],
};
