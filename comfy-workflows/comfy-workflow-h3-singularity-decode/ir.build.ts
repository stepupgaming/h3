/**
 * AUTHORING SOURCE. Do not hand-edit generated workflow.ir.json / comfy.workflow.json.
 *
 * Singularity process 4 of 4. Video and audio VAE decode only. No UNET, no CLIP.
 * Rebuild: pnpm comfy:build
 */
import { unsafeRef, workflow } from "@stepupgaming/comfy-workflows";
import type { Graph } from "@stepupgaming/comfy-workflows";
import {
  CreateVideo,
  SaveVideo,
  VAEDecode,
  VAEDecodeAudio,
  VAELoader,
} from "../environments/h3/nodes/registry.ts";

export function buildTemplate(): Graph {
  const g = workflow("comfy-workflow-h3-singularity-decode");
  const video_vae = g.param("video_vae", { type: "string", default: "minimax_h3_video_vae_fp16.safetensors" });
  const audio_vae = g.param("audio_vae", { type: "string", default: "minimax_h3_audio_vae_fp32.safetensors" });
  const latent_path = g.param("latent_path", { type: "string" });
  const output_prefix = g.param("output_prefix", { type: "string", default: "h3/singularity_dual" });

  const loaded = g.rawNode(
    "GemmyH3LoadAVLatent",
    { path: latent_path },
    {
      outputs: [
        { name: "latent", type: "LATENT" },
        { name: "fingerprint", type: "STRING" },
      ],
      id: "1",
    },
  );
  const videoVae = g.add(VAELoader, { vae_name: video_vae }, { id: "2" });
  const audioVae = g.add(VAELoader, { vae_name: audio_vae }, { id: "3" });
  const decoded = g.add(VAEDecode, {
    samples: unsafeRef(loaded.id, 0),
    vae: unsafeRef(videoVae.id, 0),
  }, { id: "4" });
  const decodedAudio = g.add(VAEDecodeAudio, {
    samples: unsafeRef(loaded.id, 0),
    vae: unsafeRef(audioVae.id, 0),
  }, { id: "5" });
  const video = g.add(CreateVideo, {
    fps: 24,
    images: unsafeRef(decoded.id, 0),
    audio: unsafeRef(decodedAudio.id, 0),
  }, { id: "6" });
  const saved = g.add(SaveVideo, {
    filename_prefix: output_prefix,
    format: "auto",
    codec: "auto" as never,
    video: unsafeRef(video.id, 0),
  }, { id: "7" });
  g.output(saved.out(0));
  return g.toGraph();
}

export default buildTemplate;

export const manifestMeta = {
  name: "comfy-workflow-h3-singularity-decode",
  title: "H3 Singularity decode",
  description: "Singularity process 4: video and audio VAE decode. No UNET.",
  outputs: [{ name: "output-0", type: "IMAGE" }],
  models: [
    { kind: "model", name: "minimax_h3_video_vae_fp16.safetensors" },
    { kind: "model", name: "minimax_h3_audio_vae_fp32.safetensors" },
  ],
  environment: "h3",
  nodePacks: ["gemmy-h3-context"],
};
