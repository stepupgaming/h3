/**
 * AUTHORING SOURCE. Do not hand-edit generated workflow.ir.json / comfy.workflow.json.
 *
 * Singularity process 2 of 4. UNET + turbo LoRA. Primary 2 steps, 1.25x upscale,
 * middle 0-step pass. No CLIP, no VAE. Saves the low sigmas for process 3.
 * Rebuild: pnpm comfy:build
 */
import { unsafeRef, workflow } from "@stepupgaming/comfy-workflows";
import type { Graph } from "@stepupgaming/comfy-workflows";
import {
  BasicGuider,
  BasicScheduler,
  ExtendIntermediateSigmas,
  GemmyH3SaveAVLatent,
  KSamplerSelect,
  LTXVConcatAVLatent,
  LTXVSeparateAVLatent,
  MinimaxH3LatentUpscaler3D,
  RandomNoise,
  SamplerCustomAdvanced,
  SplitSigmas,
} from "../environments/h3/nodes/registry.ts";
import { addSingularityUnet } from "../src/singularity.ts";

export function buildTemplate(): Graph {
  const g = workflow("comfy-workflow-h3-singularity-turbo");
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
  const seed = g.param("seed", { type: "int", default: 42 });
  const conditioning_path = g.param("conditioning_path", { type: "string" });
  const latent_path = g.param("latent_path", { type: "string" });
  const upscale_model = g.param("upscale_model", {
    type: "string",
    default: "minimax_h3_latent_upscaler_3d_bf16.safetensors",
  });
  const upscale_mode = g.param("upscale_mode", { type: "string", default: "scale by multiplier" });
  const upscale_scale = g.param("upscale_scale", { type: "float", default: 1.25 });
  const latent_prefix = g.param("latent_prefix", { type: "string", default: "h3/singularity_turbo" });
  const sigmas_prefix = g.param("sigmas_prefix", { type: "string", default: "h3/singularity_sig" });
  const fingerprint = g.param("fingerprint", { type: "string", default: "{}" });

  const model = addSingularityUnet(g, {
    unet,
    turboLora: turbo_lora,
    lmsLora: lms_lora,
    realismLora: realism_lora,
    tau,
    stack: "turbo",
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
  const primaryNoise = g.add(RandomNoise, { noise_seed: seed }, { id: "12" });
  const primaryGuider = g.add(BasicGuider, {
    model,
    conditioning: unsafeRef(cond.id, 0),
  }, { id: "13" });
  const euler = g.add(KSamplerSelect, { sampler_name: "euler" }, { id: "14" });
  const scheduler = g.add(BasicScheduler, {
    scheduler: "simple",
    steps: 6,
    denoise: 1,
    model,
  }, { id: "15" });
  const extended = g.add(ExtendIntermediateSigmas, {
    sigmas: unsafeRef(scheduler.id, 0),
    steps: 2,
    start_at_sigma: 1,
    end_at_sigma: 0,
    spacing: "linear",
  }, { id: "16" });
  const splitPrimary = g.add(SplitSigmas, {
    sigmas: unsafeRef(extended.id, 0),
    step: 2,
  }, { id: "17" });
  const splitSecond = g.add(SplitSigmas, {
    sigmas: unsafeRef(splitPrimary.id, 1),
    step: 0,
  }, { id: "18" });
  const primary = g.add(SamplerCustomAdvanced, {
    noise: unsafeRef(primaryNoise.id, 0),
    guider: unsafeRef(primaryGuider.id, 0),
    sampler: unsafeRef(euler.id, 0),
    sigmas: unsafeRef(splitPrimary.id, 0),
    latent_image: unsafeRef(loaded.id, 0),
  }, { id: "19" });
  const primaryDenoised = g.add(LTXVSeparateAVLatent, {
    av_latent: unsafeRef(primary.id, 1),
  }, { id: "20" });
  const primarySampled = g.add(LTXVSeparateAVLatent, {
    av_latent: unsafeRef(primary.id, 0),
  }, { id: "21" });
  const up = g.add(MinimaxH3LatentUpscaler3D, {
    latent: unsafeRef(primaryDenoised.id, 0),
    model_name: upscale_model,
    mode: upscale_mode,
    align: 32,
    keep_proportion: true,
    device: "cuda",
    precision: "bf16",
  }, { id: "22" });
  g.setParamRaw(up, "mode.scale", upscale_scale as never);
  const secondGuider = g.add(BasicGuider, {
    model,
    conditioning: unsafeRef(cond.id, 0),
  }, { id: "23" });
  const second = g.add(SamplerCustomAdvanced, {
    noise: unsafeRef(primaryNoise.id, 0),
    guider: unsafeRef(secondGuider.id, 0),
    sampler: unsafeRef(euler.id, 0),
    sigmas: unsafeRef(splitSecond.id, 0),
    latent_image: unsafeRef(up.id, 0),
  }, { id: "24" });
  const joined = g.add(LTXVConcatAVLatent, {
    video_latent: unsafeRef(second.id, 0),
    audio_latent: unsafeRef(primarySampled.id, 1),
  }, { id: "25" });
  const saved = g.add(GemmyH3SaveAVLatent, {
    filename_prefix: latent_prefix,
    fingerprint,
    samples: unsafeRef(joined.id, 0),
  }, { id: "26" });
  const sigmas = g.rawNode(
    "GemmyH3SaveSigmas",
    {
      sigmas: unsafeRef(splitPrimary.id, 1),
      filename_prefix: sigmas_prefix,
    },
    { outputs: [{ name: "path", type: "STRING" }], id: "27" },
  );
  g.output(saved.out(0));
  g.output(sigmas.out(0));
  return g.toGraph();
}

export default buildTemplate;

export const manifestMeta = {
  name: "comfy-workflow-h3-singularity-turbo",
  title: "H3 Singularity turbo sample",
  description: "Singularity process 2: UNET and turbo LoRA. 2 denoise steps, then exit.",
  outputs: [
    { name: "output-0", type: "STRING" },
    { name: "output-1", type: "STRING" },
  ],
  models: [
    { kind: "model", name: "Minimax-h3_Singularity_ref2va_v1.3_int8.safetensors" },
    { kind: "model", name: "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors" },
    { kind: "model", name: "minimax_h3_latent_upscaler_3d_bf16.safetensors" },
  ],
  environment: "h3",
  nodePacks: ["gemmy-h3-context", "Comfyui_Minimax_h3_latent_Upscaler"],
};
