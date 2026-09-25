/**
 * AUTHORING SOURCE. Do not hand-edit generated workflow.ir.json / comfy.workflow.json.
 *
 * Native-guide continue on Ref2VA: live still (optional Apply RefMods) plus
 * previous AV tail as keyframes. Rebuild: pnpm comfy:build
 */
import { workflow } from "@stepupgaming/comfy-workflows";
import { unsafeRef } from "@stepupgaming/comfy-workflows";
import type { Graph, NodeOutput } from "@stepupgaming/comfy-workflows";
import { addH3RefmodApply, addLatentTailGuide } from "../src/h3.ts";
import {
  CLIPLoader,
  ConditioningZeroOut,
  CreateVideo,
  GemmyH3LoadAVLatent,
  GemmyH3SaveAVLatent,
  KSampler,
  LoadImage,
  MiniMaxH3ReferenceToVideo,
  MiniMaxH3SigmaShift,
  SaveVideo,
  UNETLoader,
  VAEDecode,
  VAEDecodeAudio,
  VAELoader,
} from "../environments/h3/nodes/registry.ts";

export function buildTemplate(withRefmod = false): Graph {
  const g = workflow(
    `comfy-workflow-h3-continue-native-ref2va${withRefmod ? "-refmod" : ""}`,
  );
  const unet = g.param("unet", {
    type: "string",
    default: "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
  });
  const clip = g.param("clip", {
    type: "string",
    default: "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
  });
  const video_vae = g.param("video_vae", {
    type: "string",
    default: "minimax_h3_video_vae_fp16.safetensors",
  });
  const audio_vae = g.param("audio_vae", {
    type: "string",
    default: "minimax_h3_audio_vae_fp32.safetensors",
  });
  const ref_image = g.param("ref_image", { type: "combo" });
  const prompt = g.param("prompt", { type: "string" });
  const width = g.param("width", { type: "int", default: 864 });
  const height = g.param("height", { type: "int", default: 480 });
  const length = g.param("length", { type: "int", default: 124 });
  const seed = g.param("seed", { type: "int", default: 42 });
  const steps = g.param("steps", { type: "int", default: 20 });
  const denoise = g.param("denoise", { type: "float", default: 1 });
  const shift_video = g.param("shift_video", { type: "int", default: 12 });
  const shift_audio = g.param("shift_audio", { type: "int", default: 3 });
  const context_latent = g.param("context_latent", { type: "string" });
  const context_frames = g.param("context_frames", { type: "int", default: 39 });
  const latent_prefix = g.param("latent_prefix", { type: "string", default: "h3/continue_av" });
  const output_prefix = g.param("output_prefix", { type: "string", default: "h3/continue" });
  const fingerprint = g.param("fingerprint", { type: "string", default: "{}" });

  const unetLoader = g.add(UNETLoader, { unet_name: unet, weight_dtype: "default" }, { id: "1" });
  const clipLoader = g.add(
    CLIPLoader,
    { clip_name: clip, type: "minimax", device: "cpu" },
    { id: "2" },
  );
  const videoVae = g.add(VAELoader, { vae_name: video_vae }, { id: "3" });
  const audioVae = g.add(VAELoader, { vae_name: audio_vae }, { id: "4" });
  const shift = g.add(
    MiniMaxH3SigmaShift,
    {
      shift_video,
      shift_audio,
      model: unsafeRef(unetLoader.id, 0),
    },
    { id: "5" },
  );
  const loadImage = g.add(LoadImage, { image: ref_image }, { id: "6" });
  const ref2va = g.add(
    MiniMaxH3ReferenceToVideo,
    {
      prompt,
      width,
      height,
      length,
      ref_image_size: "match",
      clip: unsafeRef(clipLoader.id, 0),
      vae: unsafeRef(videoVae.id, 0),
      audio_vae: unsafeRef(audioVae.id, 0),
    },
    { id: "7" },
  );
  g.connectInput(ref2va, "ref_images.ref_image_0", unsafeRef(loadImage.id, 0));
  let cond: NodeOutput<"CONDITIONING"> = unsafeRef(ref2va.id, 0) as NodeOutput<"CONDITIONING">;
  if (withRefmod) {
    cond = addH3RefmodApply(g, cond);
  }
  const loadAv = g.add(GemmyH3LoadAVLatent, { path: context_latent }, { id: "16" });
  const tailGuide = addLatentTailGuide(g, {
    positive: cond,
    previous: unsafeRef(loadAv.id, 0),
    latent: unsafeRef(ref2va.id, 1),
    contextFrames: context_frames,
    id: "17",
  });
  const zero = g.add(
    ConditioningZeroOut,
    { conditioning: unsafeRef(tailGuide.id, 0) },
    { id: "8" },
  );
  const sampler = g.add(
    KSampler,
    {
      seed,
      steps,
      cfg: 1,
      sampler_name: "euler",
      scheduler: "simple",
      denoise,
      model: unsafeRef(shift.id, 0),
      positive: unsafeRef(tailGuide.id, 0),
      negative: unsafeRef(zero.id, 0),
      latent_image: unsafeRef(ref2va.id, 1),
    },
    { id: "9" },
  );
  const saveLatent = g.add(
    GemmyH3SaveAVLatent,
    {
      filename_prefix: latent_prefix,
      fingerprint,
      samples: unsafeRef(sampler.id, 0),
    },
    { id: "10" },
  );
  const vdec = g.add(
    VAEDecode,
    { samples: unsafeRef(sampler.id, 0), vae: unsafeRef(videoVae.id, 0) },
    { id: "11" },
  );
  const adec = g.add(
    VAEDecodeAudio,
    { samples: unsafeRef(sampler.id, 0), vae: unsafeRef(audioVae.id, 0) },
    { id: "12" },
  );
  const video = g.add(
    CreateVideo,
    { fps: 24, images: unsafeRef(vdec.id, 0), audio: unsafeRef(adec.id, 0) },
    { id: "13" },
  );
  const save = g.add(
    SaveVideo,
    {
      filename_prefix: output_prefix,
      format: "auto",
      codec: "auto" as never,
      video: unsafeRef(video.id, 0),
    },
    { id: "14" },
  );
  g.output(saveLatent.out(0));
  g.output(save.out(0));
  return g.toGraph();
}

export default buildTemplate;

export const manifestMeta = {
  name: "comfy-workflow-h3-continue-native-ref2va",
  title: "H3 Native-Guide Continue (Ref2VA still)",
  description:
    "Native-guide continue with one live --ref-image. Default UNET is Ref2VA (runtime binds Eros).",
  outputs: [
    { name: "output-0", type: "IMAGE" },
    { name: "output-1", type: "IMAGE" },
  ],
  models: [
    { kind: "model", name: "minimax_h3_ref2va_pruned_int8_convrot.safetensors" },
    { kind: "model", name: "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" },
    { kind: "model", name: "minimax_h3_video_vae_fp16.safetensors" },
    { kind: "model", name: "minimax_h3_audio_vae_fp32.safetensors" },
  ],
  environment: "h3",
  nodePacks: ["gemmy-h3-context"],
};
