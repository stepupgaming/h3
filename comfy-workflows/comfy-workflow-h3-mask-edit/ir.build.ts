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
  GemmyH3EncodeVideoFrames,
  GemmyH3JoinAV,
  GemmyH3SaveAVLatent,
  GemmyH3SetAVNoiseMask,
  GemmyH3SplitAV,
  GetVideoComponents,
  ImageToMask,
  KSampler,
  LoadVideo,
  MiniMaxH3ImageToVideo,
  MiniMaxH3SigmaShift,
  SaveVideo,
  UNETLoader,
  VAEDecode,
  VAEDecodeAudio,
  VAELoader,
} from "../environments/h3/nodes/registry.ts";

export function buildTemplate(): Graph {
  const g = workflow("comfy-workflow-h3-mask-edit");
  const unet = g.param("unet", { type: "string", default: "minimax_h3_fl2va_pruned_int8_convrot.safetensors" });
  const clip = g.param("clip", { type: "string", default: "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" });
  const video_vae = g.param("video_vae", { type: "string", default: "minimax_h3_video_vae_fp16.safetensors" });
  const audio_vae = g.param("audio_vae", { type: "string", default: "minimax_h3_audio_vae_fp32.safetensors" });
  const crop_video = g.param("crop_video", { type: "string" });
  const crop_mask_video = g.param("crop_mask_video", { type: "string" });
  const prompt = g.param("prompt", { type: "string" });
  const width = g.param("width", { type: "int", default: 864 });
  const height = g.param("height", { type: "int", default: 480 });
  const length = g.param("length", { type: "int", default: 124 });
  const seed = g.param("seed", { type: "int", default: 42 });
  const steps = g.param("steps", { type: "int", default: 20 });
  const denoise = g.param("denoise", { type: "float", default: 0.65 });
  const shift_video = g.param("shift_video", { type: "int", default: 12 });
  const shift_audio = g.param("shift_audio", { type: "int", default: 3 });
  const latent_prefix = g.param("latent_prefix", { type: "string", default: "h3/mask_edit_av" });
  const output_prefix = g.param("output_prefix", { type: "string", default: "h3/mask_edit" });
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
  const loadVideo_7 = g.add(LoadVideo, {
    "file": crop_video,
  }, { id: "7" });
  const getVideoComponents_8 = g.add(GetVideoComponents, {
    "video": unsafeRef(loadVideo_7.id, 0),
  }, { id: "8" });
  const gemmyH3EncodeVideoFrames_9 = g.add(GemmyH3EncodeVideoFrames, {
    "images": unsafeRef(getVideoComponents_8.id, 0),
    "vae": unsafeRef(vaeLoader_3.id, 0),
  }, { id: "9" });
  const gemmyH3SplitAV_10 = g.add(GemmyH3SplitAV, {
    "samples": unsafeRef(miniMaxH3ImageToVideo_6.id, 1),
  }, { id: "10" });
  const gemmyH3JoinAV_11 = g.add(GemmyH3JoinAV, {
    "video": unsafeRef(gemmyH3EncodeVideoFrames_9.id, 0),
    "audio": unsafeRef(gemmyH3SplitAV_10.id, 1),
  }, { id: "11" });
  const loadVideo_12 = g.add(LoadVideo, {
    "file": crop_mask_video,
  }, { id: "12" });
  const getVideoComponents_13 = g.add(GetVideoComponents, {
    "video": unsafeRef(loadVideo_12.id, 0),
  }, { id: "13" });
  const imageToMask_14 = g.add(ImageToMask, {
    "channel": "red",
    "image": unsafeRef(getVideoComponents_13.id, 0),
  }, { id: "14" });
  const gemmyH3SetAVNoiseMask_15 = g.add(GemmyH3SetAVNoiseMask, {
    "samples": unsafeRef(gemmyH3JoinAV_11.id, 0),
    "mask": unsafeRef(imageToMask_14.id, 0),
  }, { id: "15" });
  const conditioningZeroOut_16 = g.add(ConditioningZeroOut, {
    "conditioning": unsafeRef(miniMaxH3ImageToVideo_6.id, 0),
  }, { id: "16" });
  const kSampler_17 = g.add(KSampler, {
    "seed": seed,
    "steps": steps,
    "cfg": 1,
    "sampler_name": "euler",
    "scheduler": "simple",
    "denoise": denoise,
    "model": unsafeRef(miniMaxH3SigmaShift_5.id, 0),
    "positive": unsafeRef(miniMaxH3ImageToVideo_6.id, 0),
    "negative": unsafeRef(conditioningZeroOut_16.id, 0),
    "latent_image": unsafeRef(gemmyH3SetAVNoiseMask_15.id, 0),
  }, { id: "17" });
  const gemmyH3SaveAVLatent_18 = g.add(GemmyH3SaveAVLatent, {
    "filename_prefix": latent_prefix,
    "fingerprint": fingerprint,
    "samples": unsafeRef(kSampler_17.id, 0),
  }, { id: "18" });
  const vaeDecode_19 = g.add(VAEDecode, {
    "samples": unsafeRef(kSampler_17.id, 0),
    "vae": unsafeRef(vaeLoader_3.id, 0),
  }, { id: "19" });
  const vaeDecodeAudio_20 = g.add(VAEDecodeAudio, {
    "samples": unsafeRef(kSampler_17.id, 0),
    "vae": unsafeRef(vaeLoader_4.id, 0),
  }, { id: "20" });
  const createVideo_21 = g.add(CreateVideo, {
    "fps": 24,
    "images": unsafeRef(vaeDecode_19.id, 0),
    "audio": unsafeRef(vaeDecodeAudio_20.id, 0),
  }, { id: "21" });
  const saveVideo_22 = g.add(SaveVideo, {
    "filename_prefix": output_prefix,
    "format": "auto",
    "codec": "auto" as never,
    "video": unsafeRef(createVideo_21.id, 0),
  }, { id: "22" });
  g.output(gemmyH3SaveAVLatent_18.out(0));
  g.output(saveVideo_22.out(0));
  return g.toGraph();
}

export default buildTemplate;

export const manifestMeta = {
  name: "comfy-workflow-h3-mask-edit",
  title: "H3 Masked Inpaint",
  description: "H3 mask_edit continuation: encode crop video, apply mask, sample. Used after SAM3 track.",
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
