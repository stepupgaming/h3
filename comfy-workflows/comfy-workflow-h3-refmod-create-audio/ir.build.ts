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
  const g = workflow("comfy-workflow-h3-refmod-create-audio");
  const audio_vae = g.param("audio_vae", {
    type: "string",
    default: "minimax_h3_audio_vae_fp32.safetensors",
  });
  const audio = g.param("audio", { type: "combo" });
  const name = g.param("name", { type: "string", default: "voice" });
  const max_seconds = g.param("max_seconds", { type: "float", default: 15 });
  const max_tokens = g.param("max_tokens", { type: "int", default: 5120 });
  const concept_type = g.param("concept_type", { type: "string", default: "voice" });
  const subfolder = g.param("subfolder", { type: "string", default: "" });
  const description = g.param("description", { type: "string", default: "" });

  const vae = g.add(VAELoader, { vae_name: audio_vae }, { id: "1" });
  const load = g.add(LoadAudio, { audio }, { id: "2" });
  const extract = g.rawNode(
    "MiniMaxH3RefModAudioExtract",
    {
      audio: unsafeRef(load.id, 0),
      audio_vae: unsafeRef(vae.id, 0),
      name,
      max_seconds,
      max_tokens,
      budget_policy: "error",
      concept_type,
      description,
      subfolder,
      save: false,
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
  name: "comfy-workflow-h3-refmod-create-audio",
  title: "H3 RefMod create (audio)",
  description:
    "Wav/mp3 → MiniMax H3 audio VAE encode → saved audio RefMod. Experimental; speaker identity transfer is not claimed. Gemmy `video h3 refmod create --audio`.",
  outputs: [{ name: "output-0", type: "STRING" }],
  models: [{ kind: "vae", name: "minimax_h3_audio_vae_fp32.safetensors" }],
  environment: "h3",
  nodePacks: ["ComfyUI-MiniMaxH3Mod"],
};
