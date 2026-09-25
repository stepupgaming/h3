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
  GetVideoComponents,
  H3FaceStitch,
  H3FaceTrackCrop,
  H3InjectVideoLatent,
  KSampler,
  LoadVideo,
  MiniMaxH3ImageToVideo,
  MiniMaxH3SigmaShift,
  SaveVideo,
  UNETLoader,
  VAEDecode,
  VAELoader,
} from "../environments/h3/nodes/registry.ts";

export function buildTemplate(): Graph {
  const g = workflow("comfy-workflow-h3-face-refine");
  const video = g.param("video", { type: "string" });
  const detector = g.param("detector", { type: "string", default: "face_yolov8m.pt" });
  const unet = g.param("unet", { type: "string", default: "minimax_h3_fl2va_pruned_int8_convrot.safetensors" });
  const clip = g.param("clip", { type: "string", default: "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" });
  const video_vae = g.param("video_vae", { type: "string", default: "minimax_h3_video_vae_fp16.safetensors" });
  const prompt = g.param("prompt", { type: "string" });
  const length = g.param("length", { type: "int", default: 124 });
  const seed = g.param("seed", { type: "int", default: 42 });
  const steps = g.param("steps", { type: "int", default: 8 });
  const denoise = g.param("denoise", { type: "float", default: 0.35 });
  const output_prefix = g.param("output_prefix", { type: "string", default: "h3/face" });

  const loadVideo_1 = g.add(LoadVideo, {
    "file": video,
  }, { id: "1" });
  const getVideoComponents_2 = g.add(GetVideoComponents, {
    "video": unsafeRef(loadVideo_1.id, 0),
  }, { id: "2" });
  const h3FaceTrackCrop_3 = g.add(H3FaceTrackCrop, {
    "detector": detector,
    "confidence": 0.35,
    "crop_factor": 2.5,
    "canvas_width": 512,
    "canvas_height": 512,
    "canvas_mode": "auto_capped_768",
    "smooth_window": 21,
    "size_smooth_window": 51,
    "smooth_method": "gaussian",
    "size_mode": "per_frame",
    "images": unsafeRef(getVideoComponents_2.id, 0),
  }, { id: "3" });
  const unetLoader_4 = g.add(UNETLoader, {
    "unet_name": unet,
    "weight_dtype": "default",
  }, { id: "4" });
  const clipLoader_5 = g.add(CLIPLoader, {
    "clip_name": clip,
    "type": "minimax",
    "device": "cpu",
  }, { id: "5" });
  const vaeLoader_6 = g.add(VAELoader, {
    "vae_name": video_vae,
  }, { id: "6" });
  const miniMaxH3SigmaShift_7 = g.add(MiniMaxH3SigmaShift, {
    "shift_video": 12,
    "shift_audio": 3,
    "model": unsafeRef(unetLoader_4.id, 0),
  }, { id: "7" });
  const miniMaxH3ImageToVideo_8 = g.add(MiniMaxH3ImageToVideo, {
    "prompt": prompt,
    "width": 512,
    "height": 512,
    "length": length,
    "clip": unsafeRef(clipLoader_5.id, 0),
    "vae": unsafeRef(vaeLoader_6.id, 0),
  }, { id: "8" });
  g.connectInput(miniMaxH3ImageToVideo_8, "width", unsafeRef(h3FaceTrackCrop_3.id, 4));
  g.connectInput(miniMaxH3ImageToVideo_8, "height", unsafeRef(h3FaceTrackCrop_3.id, 5));
  const h3InjectVideoLatent_9 = g.add(H3InjectVideoLatent, {
    "av_latent": unsafeRef(miniMaxH3ImageToVideo_8.id, 1),
    "images": unsafeRef(h3FaceTrackCrop_3.id, 0),
    "vae": unsafeRef(vaeLoader_6.id, 0),
  }, { id: "9" });
  const conditioningZeroOut_10 = g.add(ConditioningZeroOut, {
    "conditioning": unsafeRef(miniMaxH3ImageToVideo_8.id, 0),
  }, { id: "10" });
  const kSampler_11 = g.add(KSampler, {
    "seed": seed,
    "steps": steps,
    "cfg": 1,
    "sampler_name": "euler",
    "scheduler": "simple",
    "denoise": denoise,
    "model": unsafeRef(miniMaxH3SigmaShift_7.id, 0),
    "positive": unsafeRef(miniMaxH3ImageToVideo_8.id, 0),
    "negative": unsafeRef(conditioningZeroOut_10.id, 0),
    "latent_image": unsafeRef(h3InjectVideoLatent_9.id, 0),
  }, { id: "11" });
  const vaeDecode_12 = g.add(VAEDecode, {
    "samples": unsafeRef(kSampler_11.id, 0),
    "vae": unsafeRef(vaeLoader_6.id, 0),
  }, { id: "12" });
  const h3FaceStitch_13 = g.add(H3FaceStitch, {
    "paste_region": "face_only",
    "mask_dilation": 16,
    "feather": 6,
    "colour_match": 1,
    "blend": 1,
    "undetected_frames": "fade_out",
    "base_images": unsafeRef(getVideoComponents_2.id, 0),
    "refined_crops": unsafeRef(vaeDecode_12.id, 0),
    "transform": unsafeRef(h3FaceTrackCrop_3.id, 1),
  }, { id: "13" });
  const createVideo_14 = g.add(CreateVideo, {
    "fps": 24,
    "images": unsafeRef(h3FaceStitch_13.id, 0),
    "audio": unsafeRef(getVideoComponents_2.id, 1),
  }, { id: "14" });
  const saveVideo_15 = g.add(SaveVideo, {
    "filename_prefix": output_prefix,
    "format": "auto",
    "codec": "auto" as never,
    "video": unsafeRef(createVideo_14.id, 0),
  }, { id: "15" });
  g.output(saveVideo_15.out(0));
  return g.toGraph();
}

export default buildTemplate;

export const manifestMeta = {
  name: "comfy-workflow-h3-face-refine",
  title: "H3 Face Refine",
  description: "Track/crop faces, H3 denoise 0.35, stitch back. Gemmy `video h3 face-refine`.",
  outputs: [
  {
    "name": "output-0",
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
  }
],
  environment: "h3",
  nodePacks: ["ComfyUI-H3-FaceRefine"],
};
