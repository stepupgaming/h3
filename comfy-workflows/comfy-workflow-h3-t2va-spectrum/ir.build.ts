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
  MiniMaxH3ImageToVideo,
  MiniMaxH3SigmaShift,
  SaveVideo,
  SpectrumApplyMiniMaxH3,
  UNETLoader,
  VAEDecode,
  VAEDecodeAudio,
  VAELoader,
} from "../environments/h3/nodes/registry.ts";

export function buildTemplate(): Graph {
  const g = workflow("comfy-workflow-h3-t2va-spectrum");
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
  const latent_prefix = g.param("latent_prefix", { type: "string", default: "h3/t2va_spectrum_av" });
  const fingerprint = g.param("fingerprint", { type: "string", default: "{}" });
  const output_prefix = g.param("output_prefix", { type: "string", default: "h3/t2va_spectrum" });

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
  const spectrumApplyMiniMaxH3_5 = g.add(SpectrumApplyMiniMaxH3, {
    "enabled": true,
    "blend_weight": 0.5,
    "degree": 1,
    "ridge_lambda": 0.1,
    "window_size": 2,
    "flex_window": 0.75,
    "warmup_steps": 1,
    "tail_actual_steps": 1,
    "max_history": 8,
    "debug": false,
    "history_storage": "system_ram",
    "bootstrap_first_forecast": true,
    "anchor_residual_feedback": false,
    "selective_rollback_correction": false,
    "offline_smoothing_replay": true,
    "audio_blend_weight": 0,
    "offline_archive_storage": "system_ram",
    "model": unsafeRef(unetLoader_1.id, 0),
  }, { id: "5" });
  const miniMaxH3SigmaShift_6 = g.add(MiniMaxH3SigmaShift, {
    "shift_video": shift_video,
    "shift_audio": shift_audio,
    "model": unsafeRef(spectrumApplyMiniMaxH3_5.id, 0),
  }, { id: "6" });
  const miniMaxH3ImageToVideo_7 = g.add(MiniMaxH3ImageToVideo, {
    "prompt": prompt,
    "width": width,
    "height": height,
    "length": length,
    "clip": unsafeRef(clipLoader_2.id, 0),
    "vae": unsafeRef(vaeLoader_3.id, 0),
  }, { id: "7" });
  const conditioningZeroOut_8 = g.add(ConditioningZeroOut, {
    "conditioning": unsafeRef(miniMaxH3ImageToVideo_7.id, 0),
  }, { id: "8" });
  const kSampler_9 = g.add(KSampler, {
    "seed": seed,
    "steps": steps,
    "cfg": 1,
    "sampler_name": "euler",
    "scheduler": "simple",
    "denoise": denoise,
    "model": unsafeRef(miniMaxH3SigmaShift_6.id, 0),
    "positive": unsafeRef(miniMaxH3ImageToVideo_7.id, 0),
    "negative": unsafeRef(conditioningZeroOut_8.id, 0),
    "latent_image": unsafeRef(miniMaxH3ImageToVideo_7.id, 1),
  }, { id: "9" });
  const gemmyH3SaveAVLatent_10 = g.add(GemmyH3SaveAVLatent, {
    "filename_prefix": latent_prefix,
    "fingerprint": fingerprint,
    "samples": unsafeRef(kSampler_9.id, 0),
  }, { id: "10" });
  const vaeDecode_11 = g.add(VAEDecode, {
    "samples": unsafeRef(kSampler_9.id, 0),
    "vae": unsafeRef(vaeLoader_3.id, 0),
  }, { id: "11" });
  const vaeDecodeAudio_12 = g.add(VAEDecodeAudio, {
    "samples": unsafeRef(kSampler_9.id, 0),
    "vae": unsafeRef(vaeLoader_4.id, 0),
  }, { id: "12" });
  const createVideo_13 = g.add(CreateVideo, {
    "fps": 24,
    "images": unsafeRef(vaeDecode_11.id, 0),
    "audio": unsafeRef(vaeDecodeAudio_12.id, 0),
  }, { id: "13" });
  const saveVideo_14 = g.add(SaveVideo, {
    "filename_prefix": output_prefix,
    "format": "auto",
    "codec": "auto" as never,
    "video": unsafeRef(createVideo_13.id, 0),
  }, { id: "14" });
  g.output(gemmyH3SaveAVLatent_10.out(0));
  g.output(saveVideo_14.out(0));
  return g.toGraph();
}

export default buildTemplate;

export const manifestMeta = {
  name: "comfy-workflow-h3-t2va-spectrum",
  title: "Comfy Workflow H3 T2va Spectrum",
  description: "Imported ComfyUI workflow packaged as @stepupgaming/comfy-workflow-h3-t2va-spectrum. Verify license and redistribution rights before publishing.",
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
