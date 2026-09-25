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
  CLIPLoader,
  ConditioningZeroOut,
  CreateVideo,
  GemmyH3JoinAV,
  GemmyH3LoadAVLatent,
  GemmyH3SplitAV,
  GetVideoComponents,
  KSamplerSelect,
  LoadImage,
  LoadVideo,
  ManualSigmas,
  MiniMaxH3ReferenceToVideo,
  MiniMaxH3SigmaShift,
  MinimaxH3LatentUpscaler3D,
  RandomNoise,
  SamplerCustomAdvanced,
  SaveVideo,
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
  const g = workflow(`comfy-workflow-h3-ganloss-stage2${suffix}`);
  const unet = g.param("unet", { type: "string", default: "minimax_h3_ref2va_pruned_int8_convrot.safetensors" });
  const clip = g.param("clip", { type: "string", default: "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" });
  const video_vae = g.param("video_vae", { type: "string", default: "minimax_h3_video_vae_fp16.safetensors" });
  const audio_vae = g.param("audio_vae", { type: "string", default: "minimax_h3_audio_vae_fp32.safetensors" });
  const prompt = g.param("prompt", { type: "string" });
  const width = g.param("width", { type: "int", default: 864 });
  const height = g.param("height", { type: "int", default: 480 });
  const length = g.param("length", { type: "int", default: 124 });
  const seed = g.param("seed", { type: "int", default: 42 });
  const shift_video = g.param("shift_video", { type: "int", default: 12 });
  const shift_audio = g.param("shift_audio", { type: "int", default: 3 });
  const ref_image = g.param("ref_image", { type: "combo" });
  const context_latent = g.param("context_latent", { type: "string" });
  const upscale_model = g.param("upscale_model", { type: "string", default: "minimax_h3_latent_upscaler_3d_fp16.safetensors" });
  const output_prefix = g.param("output_prefix", { type: "string", default: "h3/ganloss_s2" });

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
  if (withVideoReference) {
    const crop_video = g.param("crop_video", { type: "string" });
    const video = g.add(LoadVideo, { file: crop_video }, { id: "7" });
    const components = g.add(GetVideoComponents, { video: unsafeRef(video.id, 0) }, { id: "8" });
    g.connectInput(miniMaxH3ReferenceToVideo_9, "ref_videos.ref_video_0", components.out(0));
    g.connectInput(miniMaxH3ReferenceToVideo_9, "ref_video_audios.ref_video_audio_0", components.out(1));
  }
  const gemmyH3LoadAVLatent_10 = g.add(GemmyH3LoadAVLatent, {
    "path": context_latent,
  }, { id: "10" });
  const gemmyH3SplitAV_11 = g.add(GemmyH3SplitAV, {
    "samples": unsafeRef(gemmyH3LoadAVLatent_10.id, 0),
  }, { id: "11" });
  const minimaxH3LatentUpscaler3D_12 = g.add(MinimaxH3LatentUpscaler3D, {
    "model_name": upscale_model,
    "mode": "target dimensions" as never,
    "align": 32,
    "keep_proportion": false,
    "device": "cuda",
    "precision": "fp16",
    "latent": unsafeRef(gemmyH3SplitAV_11.id, 0),
  }, { id: "12" });
  g.setParamRaw(minimaxH3LatentUpscaler3D_12, "mode.width", width as never);
  g.setParamRaw(minimaxH3LatentUpscaler3D_12, "mode.height", height as never);
  const gemmyH3JoinAV_13 = g.add(GemmyH3JoinAV, {
    "video": unsafeRef(minimaxH3LatentUpscaler3D_12.id, 0),
    "audio": unsafeRef(gemmyH3SplitAV_11.id, 1),
  }, { id: "13" });
  let cond: NodeOutput<"CONDITIONING"> = unsafeRef(miniMaxH3ReferenceToVideo_9.id, 0) as NodeOutput<"CONDITIONING">;
  if (withRefmod) {
    cond = addH3RefmodApply(g, cond);
  }
  const conditioningZeroOut_14 = g.add(ConditioningZeroOut, {
    "conditioning": cond,
  }, { id: "14" });
  const randomNoise_15 = g.add(RandomNoise, {
    "noise_seed": seed,
  }, { id: "15" });
  const basicGuider_16 = g.add(BasicGuider, {
    "model": unsafeRef(miniMaxH3SigmaShift_5.id, 0),
    "conditioning": cond,
  }, { id: "16" });
  const manualSigmas_17 = g.add(ManualSigmas, {
    "sigmas": "0.9035, 0.6316, 0.3158, 0.0000",
  }, { id: "17" });
  const kSamplerSelect_18 = g.add(KSamplerSelect, {
    "sampler_name": "euler",
  }, { id: "18" });
  const samplerCustomAdvanced_19 = g.add(SamplerCustomAdvanced, {
    "noise": unsafeRef(randomNoise_15.id, 0),
    "guider": unsafeRef(basicGuider_16.id, 0),
    "sampler": unsafeRef(kSamplerSelect_18.id, 0),
    "sigmas": unsafeRef(manualSigmas_17.id, 0),
    "latent_image": unsafeRef(gemmyH3JoinAV_13.id, 0),
  }, { id: "19" });
  const vaeDecode_20 = g.add(VAEDecode, {
    "samples": unsafeRef(samplerCustomAdvanced_19.id, 0),
    "vae": unsafeRef(vaeLoader_3.id, 0),
  }, { id: "20" });
  const vaeDecodeAudio_21 = g.add(VAEDecodeAudio, {
    "samples": unsafeRef(samplerCustomAdvanced_19.id, 0),
    "vae": unsafeRef(vaeLoader_4.id, 0),
  }, { id: "21" });
  const createVideo_22 = g.add(CreateVideo, {
    "fps": 24,
    "images": unsafeRef(vaeDecode_20.id, 0),
    "audio": unsafeRef(vaeDecodeAudio_21.id, 0),
  }, { id: "22" });
  const saveVideo_23 = g.add(SaveVideo, {
    "filename_prefix": output_prefix,
    "format": "auto",
    "codec": "auto" as never,
    "video": unsafeRef(createVideo_22.id, 0),
  }, { id: "23" });
  g.output(conditioningZeroOut_14.out(0));
  g.output(saveVideo_23.out(0));
  return g.toGraph();
}

export default buildTemplate;

export const manifestMeta = {
  name: "comfy-workflow-h3-ganloss-stage2",
  title: "Comfy Workflow H3 Ganloss Stage2",
  description: "Imported ComfyUI workflow packaged as @stepupgaming/comfy-workflow-h3-ganloss-stage2. Verify license and redistribution rights before publishing.",
  outputs: [
  {
    "name": "output-0",
    "type": "IMAGE"
  },
  {
    "name": "output-1",
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
    "name": "minimax_h3_latent_upscaler_3d_fp16.safetensors"
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
