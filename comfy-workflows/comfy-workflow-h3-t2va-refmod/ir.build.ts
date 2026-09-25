/**
 * AUTHORING SOURCE. Do not hand-edit generated workflow.ir.json / comfy.workflow.json.
 *
 * Target environment: h3
 * Rebuild: pnpm comfy:build
 *
 * MiniMaxH3Mod classes stay rawNode until environments/h3 is recaptured.
 *
 * Text Encode presents saved refs to Qwen (`<Picture n>` / `<Video n>` / `<Audio n>`)
 * and attaches minimax_refs. Do not also Apply the same bundle.
 */
import { workflow } from "@stepupgaming/comfy-workflows";
import { unsafeRef } from "@stepupgaming/comfy-workflows";
import type { Graph } from "@stepupgaming/comfy-workflows";
import {
  CLIPLoader,
  ConditioningZeroOut,
  CreateVideo,
  EmptyMiniMaxH3LatentAV,
  GemmyH3SaveAVLatent,
  KSampler,
  MiniMaxH3SigmaShift,
  SaveVideo,
  UNETLoader,
  VAEDecode,
  VAEDecodeAudio,
  VAELoader,
} from "../environments/h3/nodes/registry.ts";

export function buildTemplate(): Graph {
  const g = workflow("comfy-workflow-h3-t2va-refmod");
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
  const prompt = g.param("prompt", { type: "string" });
  const width = g.param("width", { type: "int", default: 864 });
  const height = g.param("height", { type: "int", default: 480 });
  const length = g.param("length", { type: "int", default: 124 });
  const seed = g.param("seed", { type: "int", default: 42 });
  const steps = g.param("steps", { type: "int", default: 20 });
  const denoise = g.param("denoise", { type: "float", default: 1 });
  const shift_video = g.param("shift_video", { type: "int", default: 12 });
  const shift_audio = g.param("shift_audio", { type: "int", default: 3 });
  const latent_prefix = g.param("latent_prefix", { type: "string", default: "h3/refmod_av" });
  const output_prefix = g.param("output_prefix", { type: "string", default: "h3/refmod" });
  const fingerprint = g.param("fingerprint", { type: "string", default: "{}" });
  const none = "(none)";
  const mod1 = g.param("refmod_1", { type: "string", default: none });
  const strength1 = g.param("refmod_strength_1", { type: "float", default: 1 });
  const copies1 = g.param("refmod_copies_1", { type: "int", default: 1 });
  const mod2 = g.param("refmod_2", { type: "string", default: none });
  const strength2 = g.param("refmod_strength_2", { type: "float", default: 1 });
  const copies2 = g.param("refmod_copies_2", { type: "int", default: 1 });
  const mod3 = g.param("refmod_3", { type: "string", default: none });
  const strength3 = g.param("refmod_strength_3", { type: "float", default: 1 });
  const copies3 = g.param("refmod_copies_3", { type: "int", default: 1 });
  const mod4 = g.param("refmod_4", { type: "string", default: none });
  const strength4 = g.param("refmod_strength_4", { type: "float", default: 1 });
  const copies4 = g.param("refmod_copies_4", { type: "int", default: 1 });

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
  const loader = g.rawNode(
    "MiniMaxH3RefModsLoader",
    {
      show_info: false,
      mod_1: mod1,
      strength_1: strength1,
      copies_1: copies1,
      mod_2: mod2,
      strength_2: strength2,
      copies_2: copies2,
      mod_3: mod3,
      strength_3: strength3,
      copies_3: copies3,
      mod_4: mod4,
      strength_4: strength4,
      copies_4: copies4,
      mod_5: none,
      strength_5: 1,
      copies_5: 1,
      mod_6: none,
      strength_6: 1,
      copies_6: 1,
      mod_7: none,
      strength_7: 1,
      copies_7: 1,
      mod_8: none,
      strength_8: 1,
      copies_8: 1,
    },
    {
      outputs: [
        { name: "mods", type: "H3_REF_MODS" },
        { name: "prompt_hint", type: "STRING" },
      ],
      id: "6",
    },
  );
  const encode = g.rawNode(
    "MiniMaxH3RefModTextEncode",
    {
      clip: unsafeRef(clipLoader.id, 0),
      mods: unsafeRef(loader.id, 0),
      prompt,
      reference_fps: 24,
      max_total_tokens: 0,
      vae: unsafeRef(videoVae.id, 0),
    },
    {
      outputs: [
        { name: "conditioning", type: "CONDITIONING" },
        { name: "reference_map", type: "STRING" },
      ],
      id: "7",
    },
  );
  const empty = g.add(
    EmptyMiniMaxH3LatentAV,
    { width, height, length },
    { id: "8" },
  );
  const zero = g.add(
    ConditioningZeroOut,
    { conditioning: unsafeRef(encode.id, 0) },
    { id: "9" },
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
      positive: unsafeRef(encode.id, 0),
      negative: unsafeRef(zero.id, 0),
      latent_image: unsafeRef(empty.id, 0),
    },
    { id: "10" },
  );
  const saveLatent = g.add(
    GemmyH3SaveAVLatent,
    {
      filename_prefix: latent_prefix,
      fingerprint,
      samples: unsafeRef(sampler.id, 0),
    },
    { id: "11" },
  );
  const vdec = g.add(
    VAEDecode,
    { samples: unsafeRef(sampler.id, 0), vae: unsafeRef(videoVae.id, 0) },
    { id: "12" },
  );
  const adec = g.add(
    VAEDecodeAudio,
    { samples: unsafeRef(sampler.id, 0), vae: unsafeRef(audioVae.id, 0) },
    { id: "13" },
  );
  const video = g.add(
    CreateVideo,
    { fps: 24, images: unsafeRef(vdec.id, 0), audio: unsafeRef(adec.id, 0) },
    { id: "14" },
  );
  const save = g.add(
    SaveVideo,
    {
      filename_prefix: output_prefix,
      format: "auto",
      codec: "auto" as never,
      video: unsafeRef(video.id, 0),
    },
    { id: "15" },
  );
  g.output(saveLatent.out(0));
  g.output(save.out(0));
  g.output(encode.out(1));
  return g.toGraph();
}

export default buildTemplate;

export const manifestMeta = {
  name: "comfy-workflow-h3-t2va-refmod",
  title: "H3 generate from saved RefMods (Text Encode)",
  description:
    "Saved RefMods → H3 RefMod Text Encode (`<Picture n>` / `<Video n>`) → sample. Default UNET is Ref2VA. Gemmy `video h3 generate --ref-mod` without live stills. Do not Apply the same bundle.",
  outputs: [
    { name: "output-0", type: "IMAGE" },
    { name: "output-1", type: "IMAGE" },
    { name: "output-2", type: "STRING" },
  ],
  models: [
    { kind: "model", name: "minimax_h3_ref2va_pruned_int8_convrot.safetensors" },
    { kind: "clip", name: "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" },
    { kind: "vae", name: "minimax_h3_video_vae_fp16.safetensors" },
    { kind: "vae", name: "minimax_h3_audio_vae_fp32.safetensors" },
  ],
  environment: "h3",
  nodePacks: ["ComfyUI-MiniMaxH3Mod"],
};
