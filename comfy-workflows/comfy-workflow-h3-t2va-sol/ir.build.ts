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
  GemmyH3SaveAVLatent,
  KSampler,
  MiniMaxH3ChunkFeedForward,
  MiniMaxH3FusedModulation,
  MiniMaxH3ImageToVideo,
  MiniMaxH3MemoryEfficientSageAttentionPatch,
  MiniMaxH3ScheduledSolAttentionPatch,
  MiniMaxH3SigmaShift,
  PathchSageAttentionKJ,
  SaveVideo,
  UNETLoader,
  VAEDecode,
  VAEDecodeAudio,
  VAELoader,
} from "../environments/h3/nodes/registry.ts";

export function buildTemplate(): Graph {
  const g = workflow("comfy-workflow-h3-t2va-sol");
  const unet = g.param("unet", { type: "string", default: "minimax_h3_fl2va_pruned_int8_convrot.safetensors" });
  const clip = g.param("clip", { type: "string", default: "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" });
  const video_vae = g.param("video_vae", { type: "string", default: "minimax_h3_video_vae_fp16.safetensors" });
  const audio_vae = g.param("audio_vae", { type: "string", default: "minimax_h3_audio_vae_fp32.safetensors" });
  const prompt = g.param("prompt", { type: "string" });
  const width = g.param("width", { type: "int", default: 864 });
  const height = g.param("height", { type: "int", default: 480 });
  const length = g.param("length", { type: "int", default: 124 });
  const seed = g.param("seed", { type: "int", default: 42 });
  const steps = g.param("steps", { type: "int", default: 20 });
  const denoise = g.param("denoise", { type: "float", default: 1 });
  const shift_video = g.param("shift_video", { type: "int", default: 12 });
  const shift_audio = g.param("shift_audio", { type: "int", default: 3 });
  const latent_prefix = g.param("latent_prefix", { type: "string", default: "h3/t2va_sol_av" });
  const fingerprint = g.param("fingerprint", { type: "string", default: "{}" });
  const output_prefix = g.param("output_prefix", { type: "string", default: "h3/t2va_sol" });

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
  const pathchSageAttentionKJ_5 = g.add(PathchSageAttentionKJ, {
    "sage_attention": "auto",
    "model": unsafeRef(unetLoader_1.id, 0),
  }, { id: "5" });
  const miniMaxH3MemoryEfficientSageAttentionPatch_6 = g.add(MiniMaxH3MemoryEfficientSageAttentionPatch, {
    "model": unsafeRef(pathchSageAttentionKJ_5.id, 0),
  }, { id: "6" });
  const miniMaxH3ScheduledSolAttentionPatch_7 = g.add(MiniMaxH3ScheduledSolAttentionPatch, {
    "enabled": true,
    "tau_start": 1.3,
    "tau_end": 0.8,
    "curve": "linear",
    "min_tokens": 4096,
    "strict": false,
    "dense_percent": 0,
    "thresh_type": "diag",
    "int8_qk": false,
    "int8_pv": false,
    "sink_conditioning": "exact_kv",
    "dense_blocks": "",
    "model": unsafeRef(miniMaxH3MemoryEfficientSageAttentionPatch_6.id, 0),
  }, { id: "7" });
  const miniMaxH3FusedModulation_8 = g.add(MiniMaxH3FusedModulation, {
    "enabled": true,
    "model": unsafeRef(miniMaxH3ScheduledSolAttentionPatch_7.id, 0),
  }, { id: "8" });
  const miniMaxH3ChunkFeedForward_9 = g.add(MiniMaxH3ChunkFeedForward, {
    "enabled": true,
    "chunks": 2,
    "min_tokens": 8192,
    "model": unsafeRef(miniMaxH3FusedModulation_8.id, 0),
  }, { id: "9" });
  const miniMaxH3SigmaShift_10 = g.add(MiniMaxH3SigmaShift, {
    "shift_video": shift_video,
    "shift_audio": shift_audio,
    "model": unsafeRef(miniMaxH3ChunkFeedForward_9.id, 0),
  }, { id: "10" });
  const miniMaxH3ImageToVideo_11 = g.add(MiniMaxH3ImageToVideo, {
    "prompt": prompt,
    "width": width,
    "height": height,
    "length": length,
    "clip": unsafeRef(clipLoader_2.id, 0),
    "vae": unsafeRef(vaeLoader_3.id, 0),
  }, { id: "11" });
  const conditioningZeroOut_12 = g.add(ConditioningZeroOut, {
    "conditioning": unsafeRef(miniMaxH3ImageToVideo_11.id, 0),
  }, { id: "12" });
  const kSampler_13 = g.add(KSampler, {
    "seed": seed,
    "steps": steps,
    "cfg": 1,
    "sampler_name": "euler",
    "scheduler": "simple",
    "denoise": denoise,
    "model": unsafeRef(miniMaxH3SigmaShift_10.id, 0),
    "positive": unsafeRef(miniMaxH3ImageToVideo_11.id, 0),
    "negative": unsafeRef(conditioningZeroOut_12.id, 0),
    "latent_image": unsafeRef(miniMaxH3ImageToVideo_11.id, 1),
  }, { id: "13" });
  const gemmyH3SaveAVLatent_14 = g.add(GemmyH3SaveAVLatent, {
    "filename_prefix": latent_prefix,
    "fingerprint": fingerprint,
    "samples": unsafeRef(kSampler_13.id, 0),
  }, { id: "14" });
  const vaeDecode_15 = g.add(VAEDecode, {
    "samples": unsafeRef(kSampler_13.id, 0),
    "vae": unsafeRef(vaeLoader_3.id, 0),
  }, { id: "15" });
  const vaeDecodeAudio_16 = g.add(VAEDecodeAudio, {
    "samples": unsafeRef(kSampler_13.id, 0),
    "vae": unsafeRef(vaeLoader_4.id, 0),
  }, { id: "16" });
  const createVideo_17 = g.add(CreateVideo, {
    "fps": 24,
    "images": unsafeRef(vaeDecode_15.id, 0),
    "audio": unsafeRef(vaeDecodeAudio_16.id, 0),
  }, { id: "17" });
  const saveVideo_18 = g.add(SaveVideo, {
    "filename_prefix": output_prefix,
    "format": "auto",
    "codec": "auto" as never,
    "video": unsafeRef(createVideo_17.id, 0),
  }, { id: "18" });
  g.output(gemmyH3SaveAVLatent_14.out(0));
  g.output(saveVideo_18.out(0));
  return g.toGraph();
}

export default buildTemplate;

export const manifestMeta = {
  name: "comfy-workflow-h3-t2va-sol",
  title: "Comfy Workflow H3 T2va Sol",
  description: "Imported ComfyUI workflow packaged as @stepupgaming/comfy-workflow-h3-t2va-sol. Verify license and redistribution rights before publishing.",
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
  nodePacks: [],
};
