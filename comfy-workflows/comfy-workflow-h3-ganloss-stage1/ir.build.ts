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
import type { Graph, NodeOutput } from "@stepupgaming/comfy-workflows";
import { addH3LiveRefAudios, addH3RefmodApply } from "../src/h3.ts";
import {
  BasicGuider,
  BasicScheduler,
  CLIPLoader,
  ConditioningZeroOut,
  CreateVideo,
  GemmyH3SaveAVLatent,
  GetVideoComponents,
  KSamplerSelect,
  LoadImage,
  LoadVideo,
  MiniMaxH3ReferenceToVideo,
  MiniMaxH3SigmaShift,
  RandomNoise,
  SamplerCustomAdvanced,
  SaveVideo,
  SplitSigmas,
  UNETLoader,
  VAEDecode,
  VAEDecodeAudio,
  VAELoader,
} from "../environments/h3/nodes/registry.ts";

export function buildTemplate(
  withVideoReference = true,
  withVsa = false,
  withRefmod = false,
  nAudios = 0,
): Graph {
  const audioSuffix = nAudios >= 2 ? "-2audio" : nAudios >= 1 ? "-audio" : "";
  const suffix = `${withVideoReference ? "" : "-still"}${withVsa ? "-vsa" : ""}${withRefmod ? "-refmod" : ""}${audioSuffix}`;
  const g = workflow(`comfy-workflow-h3-ganloss-stage1${suffix}`);
  const unet = g.param("unet", { type: "string", default: "minimax_h3_ref2va_pruned_int8_convrot.safetensors" });
  const clip = g.param("clip", { type: "string", default: "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" });
  const video_vae = g.param("video_vae", { type: "string", default: "minimax_h3_video_vae_fp16.safetensors" });
  const audio_vae = g.param("audio_vae", { type: "string", default: "minimax_h3_audio_vae_fp32.safetensors" });
  const prompt = g.param("prompt", { type: "string" });
  const width = g.param("width", { type: "int", default: 512 });
  const height = g.param("height", { type: "int", default: 288 });
  const length = g.param("length", { type: "int", default: 124 });
  const seed = g.param("seed", { type: "int", default: 42 });
  const steps = g.param("steps", { type: "int", default: 8 });
  const shift_video = g.param("shift_video", { type: "int", default: 12 });
  const shift_audio = g.param("shift_audio", { type: "int", default: 3 });
  const ref_image = g.param("ref_image", { type: "combo" });
  const latent_prefix = g.param("latent_prefix", { type: "string", default: "h3/ganloss_s1_av" });
  const fingerprint = g.param("fingerprint", { type: "string", default: "{}" });
  const output_prefix = g.param("output_prefix", { type: "string", default: "h3/ganloss_s1" });

  const unetLoader_1 = g.add(UNETLoader, {
    "unet_name": unet,
    "weight_dtype": "default",
  }, { id: "1" });
  let modelRef = unsafeRef(unetLoader_1.id, 0);
  if (withVsa) {
    const gate_file = g.param("gate_file", { type: "string", default: "fasth3_vsa_gate.safetensors" });
    const sparsity = g.param("sparsity", { type: "float", default: 0.75 });
    const vsa = g.rawNode(
      "Ref2VAVSAGatePatch",
      { model: modelRef, gate_file, sparsity },
      { outputs: [{ name: "MODEL", type: "MODEL" }], id: "1b" },
    );
    modelRef = unsafeRef(vsa.id, 0);
  }
  const clipLoader_2 = g.add(CLIPLoader, {
    "clip_name": clip,
    "type": "minimax",
    "device": "cpu",
  }, { id: "2" });
  const vaeLoader_3 = g.add(VAELoader, {
    "vae_name": video_vae,
  }, { id: "3" });
  const vaeLoader_4 = g.add(VAELoader, {
    "vae_name": audio_vae,
  }, { id: "4" });
  const miniMaxH3SigmaShift_5 = g.add(MiniMaxH3SigmaShift, {
    "shift_video": shift_video,
    "shift_audio": shift_audio,
    "model": modelRef,
  }, { id: "5" });
  const loadImage_6 = g.add(LoadImage, {
    "image": ref_image,
  }, { id: "6" });
  const miniMaxH3ReferenceToVideo_9 = g.add(MiniMaxH3ReferenceToVideo, {
    "prompt": prompt,
    "width": width,
    "height": height,
    "length": length,
    "ref_image_size": "match",
    "clip": unsafeRef(clipLoader_2.id, 0),
    "vae": unsafeRef(vaeLoader_3.id, 0),
    "audio_vae": unsafeRef(vaeLoader_4.id, 0),
  }, { id: "9" });
  g.connectInput(miniMaxH3ReferenceToVideo_9, "ref_images.ref_image_0", unsafeRef(loadImage_6.id, 0));
  addH3LiveRefAudios(g, nAudios, miniMaxH3ReferenceToVideo_9);
  // Video conditioning belongs to masked editing, not the still-only Eros take.
  if (withVideoReference) {
    const crop_video = g.param("crop_video", { type: "string" });
    const video = g.add(LoadVideo, { file: crop_video }, { id: "7" });
    const components = g.add(GetVideoComponents, { video: unsafeRef(video.id, 0) }, { id: "8" });
    g.connectInput(miniMaxH3ReferenceToVideo_9, "ref_videos.ref_video_0", components.out(0));
    g.connectInput(miniMaxH3ReferenceToVideo_9, "ref_video_audios.ref_video_audio_0", components.out(1));
  }
  let cond: NodeOutput<"CONDITIONING"> = unsafeRef(miniMaxH3ReferenceToVideo_9.id, 0) as NodeOutput<"CONDITIONING">;
  if (withRefmod) {
    cond = addH3RefmodApply(g, cond);
  }
  const conditioningZeroOut_10 = g.add(ConditioningZeroOut, {
    "conditioning": cond,
  }, { id: "10" });
  const randomNoise_11 = g.add(RandomNoise, {
    "noise_seed": seed,
  }, { id: "11" });
  const basicGuider_12 = g.add(BasicGuider, {
    "model": unsafeRef(miniMaxH3SigmaShift_5.id, 0),
    "conditioning": cond,
  }, { id: "12" });
  const basicScheduler_13 = g.add(BasicScheduler, {
    "scheduler": "simple",
    "steps": steps,
    "denoise": 1,
    "model": unsafeRef(miniMaxH3SigmaShift_5.id, 0),
  }, { id: "13" });
  const splitSigmas_14 = g.add(SplitSigmas, {
    "step": 4,
    "sigmas": unsafeRef(basicScheduler_13.id, 0),
  }, { id: "14" });
  const kSamplerSelect_15 = g.add(KSamplerSelect, {
    "sampler_name": "euler",
  }, { id: "15" });
  const samplerCustomAdvanced_16 = g.add(SamplerCustomAdvanced, {
    "noise": unsafeRef(randomNoise_11.id, 0),
    "guider": unsafeRef(basicGuider_12.id, 0),
    "sampler": unsafeRef(kSamplerSelect_15.id, 0),
    "sigmas": unsafeRef(splitSigmas_14.id, 0),
    "latent_image": unsafeRef(miniMaxH3ReferenceToVideo_9.id, 1),
  }, { id: "16" });
  const gemmyH3SaveAVLatent_17 = g.add(GemmyH3SaveAVLatent, {
    "filename_prefix": latent_prefix,
    "fingerprint": fingerprint,
    "samples": unsafeRef(samplerCustomAdvanced_16.id, 1),
  }, { id: "17" });
  const vaeDecode_18 = g.add(VAEDecode, {
    "samples": unsafeRef(samplerCustomAdvanced_16.id, 1),
    "vae": unsafeRef(vaeLoader_3.id, 0),
  }, { id: "18" });
  const vaeDecodeAudio_19 = g.add(VAEDecodeAudio, {
    "samples": unsafeRef(samplerCustomAdvanced_16.id, 1),
    "vae": unsafeRef(vaeLoader_4.id, 0),
  }, { id: "19" });
  const createVideo_20 = g.add(CreateVideo, {
    "fps": 24,
    "images": unsafeRef(vaeDecode_18.id, 0),
    "audio": unsafeRef(vaeDecodeAudio_19.id, 0),
  }, { id: "20" });
  const saveVideo_21 = g.add(SaveVideo, {
    "filename_prefix": output_prefix,
    "format": "auto",
    "codec": "auto" as never,
    "video": unsafeRef(createVideo_20.id, 0),
  }, { id: "21" });
  g.output(conditioningZeroOut_10.out(0));
  g.output(gemmyH3SaveAVLatent_17.out(0));
  g.output(saveVideo_21.out(0));
  return g.toGraph();
}

export default buildTemplate;

export const manifestMeta = {
  name: "comfy-workflow-h3-ganloss-stage1",
  title: "Comfy Workflow H3 Ganloss Stage1",
  description: "Imported ComfyUI workflow packaged as @stepupgaming/comfy-workflow-h3-ganloss-stage1. Verify license and redistribution rights before publishing.",
  outputs: [
  {
    "name": "output-0",
    "type": "IMAGE"
  },
  {
    "name": "output-1",
    "type": "IMAGE"
  },
  {
    "name": "output-2",
    "type": "IMAGE"
  }
],
  models: [
  {
    "kind": "model",
    "name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
  },
  {
    "kind": "model",
    "name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
  },
  {
    "kind": "model",
    "name": "minimax_h3_video_vae_fp16.safetensors"
  },
  {
    "kind": "model",
    "name": "minimax_h3_audio_vae_fp32.safetensors"
  }
],
  environment: "h3",
  nodePacks: [],
};
