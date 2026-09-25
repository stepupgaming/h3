/**
 * AUTHORING SOURCE. Do not hand-edit generated workflow.ir.json / comfy.workflow.json.
 *
 * Singularity process 3 of 4. Fresh UNET, turbo + LMS 0.5 + realism.
 * Loads the low sigmas from process 2. 10 denoise steps. No CLIP, no VAE.
 * Rebuild: pnpm comfy:build
 */
import { unsafeRef, workflow } from "@stepupgaming/comfy-workflows";
import type { Graph } from "@stepupgaming/comfy-workflows";
import {
  BasicGuider,
  DisableNoise,
  GemmyH3SaveAVLatent,
  KSamplerSelect,
  SamplerCustomAdvanced,
} from "../environments/h3/nodes/registry.ts";
import { addSingularityUnet } from "../src/singularity.ts";

export function buildTemplate(): Graph {
  const g = workflow("comfy-workflow-h3-singularity-final");
  const unet = g.param("unet", {
    type: "string",
    default: "Minimax-h3_Singularity_ref2va_v1.3_int8.safetensors",
  });
  const turbo_lora = g.param("turbo_lora", {
    type: "string",
    default: "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
  });
  const lms_lora = g.param("lms_lora", {
    type: "string",
    default: "minimax_h3_lms_v1.0_r64.safetensors",
  });
  const realism_lora = g.param("realism_lora", {
    type: "string",
    default: "h3-realism-people-t2v-i2v-r2v.safetensors",
  });
  const tau = g.param("tau", { type: "float", default: 1.3 });
  const conditioning_path = g.param("conditioning_path", { type: "string" });
  const latent_path = g.param("latent_path", { type: "string" });
  const sigmas_path = g.param("sigmas_path", { type: "string" });
  const latent_prefix = g.param("latent_prefix", { type: "string", default: "h3/singularity_final" });
  const fingerprint = g.param("fingerprint", { type: "string", default: "{}" });

  const model = addSingularityUnet(g, {
    unet,
    turboLora: turbo_lora,
    lmsLora: lms_lora,
    realismLora: realism_lora,
    tau,
    stack: "final",
  });
  const cond = g.rawNode(
    "GemmyH3LoadConditioning",
    { path: conditioning_path },
    { outputs: [{ name: "conditioning", type: "CONDITIONING" }], id: "10" },
  );
  const loaded = g.rawNode(
    "GemmyH3LoadAVLatent",
    { path: latent_path },
    {
      outputs: [
        { name: "latent", type: "LATENT" },
        { name: "fingerprint", type: "STRING" },
      ],
      id: "11",
    },
  );
  const sigmas = g.rawNode(
    "GemmyH3LoadSigmas",
    { path: sigmas_path },
    { outputs: [{ name: "sigmas", type: "SIGMAS" }], id: "12" },
  );
  const noise = g.add(DisableNoise, {}, { id: "13" });
  const guider = g.add(BasicGuider, {
    model,
    conditioning: unsafeRef(cond.id, 0),
  }, { id: "14" });
  const euler = g.add(KSamplerSelect, { sampler_name: "euler" }, { id: "15" });
  const sampled = g.add(SamplerCustomAdvanced, {
    noise: unsafeRef(noise.id, 0),
    guider: unsafeRef(guider.id, 0),
    sampler: unsafeRef(euler.id, 0),
    sigmas: unsafeRef(sigmas.id, 0),
    latent_image: unsafeRef(loaded.id, 0),
  }, { id: "16" });
  const saved = g.add(GemmyH3SaveAVLatent, {
    filename_prefix: latent_prefix,
    fingerprint,
    samples: unsafeRef(sampled.id, 1),
  }, { id: "17" });
  g.output(saved.out(0));
  return g.toGraph();
}

export default buildTemplate;

export const manifestMeta = {
  name: "comfy-workflow-h3-singularity-final",
  title: "H3 Singularity final sample",
  description: "Singularity process 3: fresh UNET, LMS and realism, 10 denoise steps, then exit.",
  outputs: [{ name: "output-0", type: "STRING" }],
  models: [
    { kind: "model", name: "Minimax-h3_Singularity_ref2va_v1.3_int8.safetensors" },
    { kind: "model", name: "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors" },
    { kind: "model", name: "minimax_h3_lms_v1.0_r64.safetensors" },
    { kind: "model", name: "h3-realism-people-t2v-i2v-r2v.safetensors" },
  ],
  environment: "h3",
  nodePacks: ["gemmy-h3-context"],
};
