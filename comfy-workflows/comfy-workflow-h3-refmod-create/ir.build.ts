/**
 * AUTHORING SOURCE. Do not hand-edit generated workflow.ir.json / comfy.workflow.json.
 *
 * Target environment: h3
 * Generated node SDK: environments/h3/nodes
 *
 * Rebuild: pnpm comfy:build
 *
 * MiniMaxH3Mod classes stay rawNode until environments/h3 is recaptured.
 */
import { workflow } from "@stepupgaming/comfy-workflows";
import { unsafeRef } from "@stepupgaming/comfy-workflows";
import type { Graph } from "@stepupgaming/comfy-workflows";
import { VAELoader } from "../environments/h3/nodes/registry.ts";

export function buildTemplate(): Graph {
  const g = workflow("comfy-workflow-h3-refmod-create");
  const video_vae = g.param("video_vae", {
    type: "string",
    default: "minimax_h3_video_vae_fp16.safetensors",
  });
  const folder = g.param("folder", { type: "string" });
  const name = g.param("name", { type: "string", default: "character" });
  const mode = g.param("mode", { type: "string", default: "Full Reference" });
  const concept_type = g.param("concept_type", { type: "string", default: "identity" });
  const ref_resolution = g.param("ref_resolution", { type: "int", default: 1024 });
  const max_tokens = g.param("max_tokens", { type: "int", default: 8192 });
  const max_items = g.param("max_items", { type: "int", default: 32 });
  const max_frames = g.param("max_frames", { type: "int", default: 240 });
  const subfolder = g.param("subfolder", { type: "string", default: "" });
  const description = g.param("description", { type: "string", default: "" });

  const vae = g.add(VAELoader, { vae_name: video_vae }, { id: "1" });
  const folderLoader = g.rawNode(
    "MiniMaxH3RefModFolderLoader",
    { folder, max_items, max_frames, max_edge: 1024 },
    {
      outputs: [
        { name: "refs", type: "H3_REF_LIST" },
        { name: "count", type: "INT" },
      ],
      id: "2",
    },
  );
  const extract = g.rawNode(
    "MiniMaxH3RefModExtract",
    {
      name,
      mode,
      concept_type,
      refs_bundle: unsafeRef(folderLoader.id, 0),
      vae: unsafeRef(vae.id, 0),
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
      extraction_preset: "manual",
      subfolder,
      description,
      save: false,
      budget_policy: "error",
    },
    {
      outputs: [{ name: "mods", type: "H3_REF_MODS" }],
      id: "3",
    },
  );
  const save = g.rawNode(
    "MiniMaxH3RefModSave",
    {
      mods: unsafeRef(extract.id, 0),
      filename_prefix: "",
      subfolder,
    },
    {
      outputs: [
        { name: "mods", type: "H3_REF_MODS" },
        { name: "saved_paths", type: "STRING" },
      ],
      id: "4",
    },
  );
  g.output(save.out(1));
  return g.toGraph();
}

export default buildTemplate;

export const manifestMeta = {
  name: "comfy-workflow-h3-refmod-create",
  title: "H3 RefMod create (visual)",
  description:
    "Folder of stills/video → MiniMax H3 video VAE encode → saved RefMod .safetensors. Gemmy `video h3 refmod create`. No DiT.",
  outputs: [{ name: "output-0", type: "STRING" }],
  models: [{ kind: "vae", name: "minimax_h3_video_vae_fp16.safetensors" }],
  environment: "h3",
  nodePacks: ["ComfyUI-MiniMaxH3Mod"],
};
