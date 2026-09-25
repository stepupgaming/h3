/**
 * AUTHORING SOURCE. Do not hand-edit generated workflow.ir.json / comfy.workflow.json.
 *
 * Target environment: h3
 * Rebuild: pnpm comfy:build
 *
 * MiniMaxH3Mod classes stay rawNode until environments/h3 is recaptured.
 */
import { workflow } from "@stepupgaming/comfy-workflows";
import { unsafeRef } from "@stepupgaming/comfy-workflows";
import type { Graph } from "@stepupgaming/comfy-workflows";
import { LoadAudio, VAELoader } from "../environments/h3/nodes/registry.ts";

export function buildTemplate(): Graph {
  const g = workflow("comfy-workflow-h3-refmod-create-master");
  const video_vae = g.param("video_vae", {
    type: "string",
    default: "minimax_h3_video_vae_fp16.safetensors",
  });
  const audio_vae = g.param("audio_vae", {
    type: "string",
    default: "minimax_h3_audio_vae_fp32.safetensors",
  });
  const folder = g.param("folder", { type: "string" });
  const audio = g.param("audio", { type: "combo" });
  const name = g.param("name", { type: "string", default: "character" });
  const mode = g.param("mode", { type: "string", default: "Full Reference" });
  const concept_type = g.param("concept_type", { type: "string", default: "identity" });
  const ref_resolution = g.param("ref_resolution", { type: "int", default: 1024 });
  const max_tokens = g.param("max_tokens", { type: "int", default: 8192 });
  const max_items = g.param("max_items", { type: "int", default: 32 });
  const max_frames = g.param("max_frames", { type: "int", default: 240 });
  const max_seconds = g.param("max_seconds", { type: "float", default: 15 });
  const subfolder = g.param("subfolder", { type: "string", default: "" });
  const description = g.param("description", { type: "string", default: "" });
  const save_layout = g.param("save_layout", { type: "string", default: "bundle" });

  const vvae = g.add(VAELoader, { vae_name: video_vae }, { id: "1" });
  const avae = g.add(VAELoader, { vae_name: audio_vae }, { id: "2" });
  const folderLoader = g.rawNode(
    "MiniMaxH3RefModFolderLoader",
    { folder, max_items, max_frames, max_edge: 1024 },
    {
      outputs: [
        { name: "refs", type: "H3_REF_LIST" },
        { name: "count", type: "INT" },
      ],
      id: "3",
    },
  );
  const loadAudio = g.add(LoadAudio, { audio }, { id: "4" });
  const master = g.rawNode(
    "MiniMaxH3RefModMasterExtract",
    {
      name,
      mode,
      concept_type,
      refs_bundle: unsafeRef(folderLoader.id, 0),
      vae: unsafeRef(vvae.id, 0),
      audio: unsafeRef(loadAudio.id, 0),
      audio_vae: unsafeRef(avae.id, 0),
      background_retention: 0,
      ref_resolution,
      pool_h: 16,
      pool_w: 16,
      latent_frames: 16,
      identity: 0,
      merge: false,
      motion_only: false,
      multiplier: 1,
      max_tokens,
      audio_max_seconds: max_seconds,
      audio_max_tokens: 5120,
      audio_budget_policy: "error",
      audio_concept_type: "voice",
      max_total_tokens: 0,
      extraction_preset: "manual",
      subfolder,
      description,
      save: true,
      budget_policy: "error",
      save_layout,
    },
    {
      outputs: [
        { name: "mods", type: "H3_REF_MODS" },
        { name: "details", type: "STRING" },
      ],
      id: "5",
    },
  );
  g.output(master.out(1));
  return g.toGraph();
}

export default buildTemplate;

export const manifestMeta = {
  name: "comfy-workflow-h3-refmod-create-master",
  title: "H3 RefMod create (visual + audio bundle)",
  description:
    "Stills/video folder plus a voice clip → one v5 RefMod bundle. Packaging does not bind voice to the face. Gemmy `video h3 refmod create --audio`.",
  outputs: [{ name: "output-0", type: "STRING" }],
  models: [
    { kind: "vae", name: "minimax_h3_video_vae_fp16.safetensors" },
    { kind: "vae", name: "minimax_h3_audio_vae_fp32.safetensors" },
  ],
  environment: "h3",
  nodePacks: ["ComfyUI-MiniMaxH3Mod"],
};
