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
import {
  addH3JevNativeSlaPatch,
  addH3JevResMultistepSampler,
  h3JevPackageSuffix,
  usesJevResMultistep,
  usesJevSlaPatch,
  type H3JevMode,
} from "../src/h3.ts";
import {
  CLIPLoader,
  ConditioningZeroOut,
  CreateVideo,
  GemmyH3SaveAVLatent,
  KSampler,
  LoadImage,
  MiniMaxH3ReferenceToVideo,
  MiniMaxH3SigmaShift,
  SaveVideo,
  UNETLoader,
  VAEDecode,
  VAEDecodeAudio,
  VAELoader,
} from "../environments/h3/nodes/registry.ts";

export function buildTemplate(withVsa = false, withJev: H3JevMode = false): Graph {
  const g = workflow(
    `comfy-workflow-h3-ref2va-multiref${withVsa ? "-vsa" : ""}${h3JevPackageSuffix(withJev)}`,
  );
  const unet = g.param("unet", { type: "string", default: "minimax_h3_ref2va_pruned_int8_convrot.safetensors" });
  const clip = g.param("clip", { type: "string", default: "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" });
  const video_vae = g.param("video_vae", { type: "string", default: "minimax_h3_video_vae_fp16.safetensors" });
  const audio_vae = g.param("audio_vae", { type: "string", default: "minimax_h3_audio_vae_fp32.safetensors" });
  const prompt = g.param("prompt", { type: "string" });
  const width = g.param("width", { type: "int", default: 864 });
  const height = g.param("height", { type: "int", default: 480 });
  const length = g.param("length", { type: "int", default: 124 });
  const seed = g.param("seed", { type: "int", default: 42 });
  const steps = g.param("steps", { type: "int", default: 20 });
  const denoise = g.param("denoise", { type: "float", default: 1 });
  const shift_video = g.param("shift_video", { type: "int", default: 12 });
  const shift_audio = g.param("shift_audio", { type: "int", default: 3 });
  const ref_image = g.param("ref_image", { type: "combo" });
  const ref_image_1 = g.param("ref_image_1", { type: "combo" });
  const latent_prefix = g.param("latent_prefix", { type: "string", default: "h3/ref2va_multi_av" });
  const fingerprint = g.param("fingerprint", { type: "string", default: "{}" });
  const output_prefix = g.param("output_prefix", { type: "string", default: "h3/ref2va_multi" });

  const unetLoader_1 = g.add(UNETLoader, {
    "unet_name": unet,
    "weight_dtype": "default",
  }, { id: "1" });
  let modelRef: NodeOutput<"MODEL"> = unsafeRef(unetLoader_1.id, 0) as NodeOutput<"MODEL">;
  if (withVsa) {
    const gate_file = g.param("gate_file", { type: "string", default: "fasth3_vsa_gate.safetensors" });
    const sparsity = g.param("sparsity", { type: "float", default: 0.75 });
    const vsa = g.rawNode(
      "Ref2VAVSAGatePatch",
      { model: modelRef, gate_file, sparsity },
      { outputs: [{ name: "MODEL", type: "MODEL" }], id: "1b" },
    );
    modelRef = unsafeRef(vsa.id, 0) as NodeOutput<"MODEL">;
  }
  if (usesJevSlaPatch(withJev)) {
    modelRef = addH3JevNativeSlaPatch(g, modelRef, prompt);
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
  const loadImage_7 = g.add(LoadImage, {
    "image": ref_image_1,
  }, { id: "7" });
  const miniMaxH3ReferenceToVideo_8 = g.add(MiniMaxH3ReferenceToVideo, {
    "prompt": prompt,
    "width": width,
    "height": height,
    "length": length,
    "ref_image_size": "match",
    "clip": unsafeRef(clipLoader_2.id, 0),
    "vae": unsafeRef(vaeLoader_3.id, 0),
    "audio_vae": unsafeRef(vaeLoader_4.id, 0),
  }, { id: "8" });
  g.connectInput(miniMaxH3ReferenceToVideo_8, "ref_images.ref_image_0", unsafeRef(loadImage_6.id, 0));
  g.connectInput(miniMaxH3ReferenceToVideo_8, "ref_images.ref_image_1", unsafeRef(loadImage_7.id, 0));
  const conditioningZeroOut_9 = g.add(ConditioningZeroOut, {
    "conditioning": unsafeRef(miniMaxH3ReferenceToVideo_8.id, 0),
  }, { id: "9" });
  let samples: NodeOutput<"LATENT">;
  if (usesJevResMultistep(withJev)) {
    samples = addH3JevResMultistepSampler(g, {
      model: unsafeRef(miniMaxH3SigmaShift_5.id, 0) as NodeOutput<"MODEL">,
      positive: unsafeRef(miniMaxH3ReferenceToVideo_8.id, 0) as NodeOutput<"CONDITIONING">,
      latent: unsafeRef(miniMaxH3ReferenceToVideo_8.id, 1) as NodeOutput<"LATENT">,
      seed,
      steps,
      ids: { noise: "10a", guider: "10b", scheduler: "10c", select: "10d", custom: "10e" },
    });
  } else {
    const kSampler_10 = g.add(KSampler, {
      "seed": seed,
      "steps": steps,
      "cfg": 1,
      "sampler_name": "euler",
      "scheduler": "simple",
      "denoise": denoise,
      "model": unsafeRef(miniMaxH3SigmaShift_5.id, 0),
      "positive": unsafeRef(miniMaxH3ReferenceToVideo_8.id, 0),
      "negative": unsafeRef(conditioningZeroOut_9.id, 0),
      "latent_image": unsafeRef(miniMaxH3ReferenceToVideo_8.id, 1),
    }, { id: "10" });
    samples = unsafeRef(kSampler_10.id, 0) as NodeOutput<"LATENT">;
  }
  const gemmyH3SaveAVLatent_11 = g.add(GemmyH3SaveAVLatent, {
    "filename_prefix": latent_prefix,
    "fingerprint": fingerprint,
    "samples": samples,
  }, { id: "11" });
  const vaeDecode_12 = g.add(VAEDecode, {
    "samples": samples,
    "vae": unsafeRef(vaeLoader_3.id, 0),
  }, { id: "12" });
  const vaeDecodeAudio_13 = g.add(VAEDecodeAudio, {
    "samples": samples,
    "vae": unsafeRef(vaeLoader_4.id, 0),
  }, { id: "13" });
  const createVideo_14 = g.add(CreateVideo, {
    "fps": 24,
    "images": unsafeRef(vaeDecode_12.id, 0),
    "audio": unsafeRef(vaeDecodeAudio_13.id, 0),
  }, { id: "14" });
  const saveVideo_15 = g.add(SaveVideo, {
    "filename_prefix": output_prefix,
    "format": "auto",
    "codec": "auto" as never,
    "video": unsafeRef(createVideo_14.id, 0),
  }, { id: "15" });
  g.output(gemmyH3SaveAVLatent_11.out(0));
  g.output(saveVideo_15.out(0));
  return g.toGraph();
}

export default buildTemplate;

export const manifestMeta = {
  name: "comfy-workflow-h3-ref2va-multiref",
  title: "Comfy Workflow H3 Ref2va Multiref",
  description: "Imported ComfyUI workflow packaged as @stepupgaming/comfy-workflow-h3-ref2va-multiref. Verify license and redistribution rights before publishing.",
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
