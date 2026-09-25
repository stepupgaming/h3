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
import type { Graph } from "@stepupgaming/comfy-workflows";
import {
  CLIPTextEncode,
  CheckpointLoaderSimple,
  CreateVideo,
  GetVideoComponents,
  LoadVideo,
  MaskToImage,
  SAM3_TrackToMask,
  SAM3_VideoTrack,
  SaveVideo,
} from "../environments/h3/nodes/registry.ts";

export function buildTemplate(): Graph {
  const g = workflow("comfy-workflow-h3-sam3-track");
  const video = g.param("video", { type: "string" });
  const checkpoint = g.param("checkpoint", { type: "combo", default: "sam3.1_multiplex_fp16.safetensors" });
  const prompt = g.param("prompt", { type: "string" });
  const output_prefix = g.param("output_prefix", { type: "string", default: "h3/sam3" });

  const loadVideo_1 = g.add(LoadVideo, {
    "file": video,
  }, { id: "1" });
  const getVideoComponents_2 = g.add(GetVideoComponents, {
    "video": unsafeRef(loadVideo_1.id, 0),
  }, { id: "2" });
  const checkpointLoaderSimple_3 = g.add(CheckpointLoaderSimple, {
    "ckpt_name": checkpoint,
  }, { id: "3" });
  const clipTextEncode_4 = g.add(CLIPTextEncode, {
    "text": prompt,
    "clip": unsafeRef(checkpointLoaderSimple_3.id, 1),
  }, { id: "4" });
  const saM3_VideoTrack_5 = g.add(SAM3_VideoTrack, {
    "detection_threshold": 0.5,
    "max_objects": 1,
    "detect_interval": 1,
    "images": unsafeRef(getVideoComponents_2.id, 0),
    "model": unsafeRef(checkpointLoaderSimple_3.id, 0),
    "conditioning": unsafeRef(clipTextEncode_4.id, 0),
  }, { id: "5" });
  const saM3_TrackToMask_6 = g.add(SAM3_TrackToMask, {
    "object_indices": "0",
    "track_data": unsafeRef(saM3_VideoTrack_5.id, 0),
  }, { id: "6" });
  const maskToImage_7 = g.add(MaskToImage, {
    "mask": unsafeRef(saM3_TrackToMask_6.id, 0),
  }, { id: "7" });
  const createVideo_8 = g.add(CreateVideo, {
    "fps": 24,
    "images": unsafeRef(maskToImage_7.id, 0),
  }, { id: "8" });
  const saveVideo_9 = g.add(SaveVideo, {
    "filename_prefix": output_prefix,
    "format": "auto",
    "codec": "auto" as never,
    "video": unsafeRef(createVideo_8.id, 0),
  }, { id: "9" });
  g.output(saveVideo_9.out(0));
  return g.toGraph();
}

export default buildTemplate;

export const manifestMeta = {
  name: "comfy-workflow-h3-sam3-track",
  title: "SAM3 Video Track to Mask",
  description: "SAM3.1 video track → mask MP4. First stage of Gemmy `video h3 edit`.",
  outputs: [
  {
    "name": "output-0",
    "type": "IMAGE"
  }
],
  models: [
  {
    "kind": "checkpoint",
    "name": "sam3.1_multiplex_fp16.safetensors"
  }
],
  environment: "h3",
  nodePacks: [],
};
