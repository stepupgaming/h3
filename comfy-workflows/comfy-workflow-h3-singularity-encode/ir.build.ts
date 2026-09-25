/**
 * AUTHORING SOURCE. Do not hand-edit generated workflow.ir.json / comfy.workflow.json.
 *
 * Singularity process 1 of 4. CLIP + both VAEs. No UNET.
 * Rebuild: pnpm comfy:build
 */
import { unsafeRef, workflow } from "@stepupgaming/comfy-workflows";
import type { Graph } from "@stepupgaming/comfy-workflows";
import {
  CLIPLoader,
  GemmyH3SaveAVLatent,
  LoadImage,
  MiniMaxH3ReferenceToVideo,
  VAELoader,
} from "../environments/h3/nodes/registry.ts";

export function buildTemplate(): Graph {
  const g = workflow("comfy-workflow-h3-singularity-encode");
  const clip = g.param("clip", { type: "string", default: "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" });
  const video_vae = g.param("video_vae", { type: "string", default: "minimax_h3_video_vae_fp16.safetensors" });
  const audio_vae = g.param("audio_vae", { type: "string", default: "minimax_h3_audio_vae_fp32.safetensors" });
  const prompt = g.param("prompt", { type: "string" });
  const width = g.param("width", { type: "int", default: 544 });
  const height = g.param("height", { type: "int", default: 960 });
  const length = g.param("length", { type: "int", default: 124 });
  const ref_image = g.param("ref_image", { type: "combo" });
  const cond_prefix = g.param("cond_prefix", { type: "string", default: "h3/singularity_cond" });
  const latent_prefix = g.param("latent_prefix", { type: "string", default: "h3/singularity_empty" });
  const fingerprint = g.param("fingerprint", { type: "string", default: "{}" });

  const clipLoader = g.add(CLIPLoader, {
    clip_name: clip,
    type: "minimax",
    device: "cpu",
  }, { id: "1" });
  const videoVae = g.add(VAELoader, { vae_name: video_vae }, { id: "2" });
  const audioVae = g.add(VAELoader, { vae_name: audio_vae }, { id: "3" });
  const still = g.add(LoadImage, { image: ref_image }, { id: "4" });
  const ref2va = g.add(MiniMaxH3ReferenceToVideo, {
    prompt,
    width,
    height,
    length,
    ref_image_size: "max",
    clip: unsafeRef(clipLoader.id, 0),
    vae: unsafeRef(videoVae.id, 0),
    audio_vae: unsafeRef(audioVae.id, 0),
  }, { id: "5" });
  g.connectInput(ref2va, "ref_images.ref_image_0", unsafeRef(still.id, 0));
  const cond = g.rawNode(
    "GemmyH3SaveConditioning",
    {
      conditioning: unsafeRef(ref2va.id, 0),
      filename_prefix: cond_prefix,
    },
    { outputs: [{ name: "path", type: "STRING" }], id: "6" },
  );
  const latent = g.add(GemmyH3SaveAVLatent, {
    filename_prefix: latent_prefix,
    fingerprint,
    samples: unsafeRef(ref2va.id, 1),
  }, { id: "7" });
  g.output(cond.out(0));
  g.output(latent.out(0));
  return g.toGraph();
}

export default buildTemplate;

export const manifestMeta = {
  name: "comfy-workflow-h3-singularity-encode",
  title: "H3 Singularity encode",
  description: "Singularity process 1: text encoder and VAEs only. Exits before the UNET loads.",
  outputs: [
    { name: "output-0", type: "STRING" },
    { name: "output-1", type: "STRING" },
  ],
  models: [
    { kind: "model", name: "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" },
    { kind: "model", name: "minimax_h3_video_vae_fp16.safetensors" },
    { kind: "model", name: "minimax_h3_audio_vae_fp32.safetensors" },
  ],
  environment: "h3",
  nodePacks: ["gemmy-h3-context"],
};
