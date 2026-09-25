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
  CreateVideo,
  GetVideoComponents,
  LoadVideo,
  RTXVideoSuperResolution,
  SaveVideo,
} from "../environments/h3/nodes/registry.ts";

export function buildTemplate(): Graph {
  const g = workflow("comfy-workflow-h3-rtx-vsr");
  const video = g.param("video", { type: "string" });
  const quality = g.param("quality", { type: "string", default: "HIGH" });
  const scale = g.param("scale", { type: "int", default: 2 });
  const resize_type = g.param("resize_type", { type: "string", default: "scale by multiplier" });
  const target_width = g.param("target_width", { type: "int", default: 1920 });
  const target_height = g.param("target_height", { type: "int", default: 1088 });
  const keep_aspect = g.param("keep_aspect", { type: "boolean", default: false });
  const output_prefix = g.param("output_prefix", { type: "string", default: "h3/rtx_vsr" });

  const loadVideo_6 = g.add(LoadVideo, {
    "file": video,
  }, { id: "6" });
  const getVideoComponents_7 = g.add(GetVideoComponents, {
    "video": unsafeRef(loadVideo_6.id, 0),
  }, { id: "7" });
  const rtxVideoSuperResolution_1 = g.add(RTXVideoSuperResolution, {
    "quality": quality,
    "resize_type": resize_type,
    "images": unsafeRef(getVideoComponents_7.id, 0),
  }, { id: "1" });
  g.setParamRaw(rtxVideoSuperResolution_1, "resize_type.scale", scale as never);
  g.setParamRaw(rtxVideoSuperResolution_1, "resize_type.width", target_width as never);
  g.setParamRaw(rtxVideoSuperResolution_1, "resize_type.height", target_height as never);
  g.setParamRaw(rtxVideoSuperResolution_1, "resize_type.keep_aspect_ratio", keep_aspect as never);
  const createVideo_8 = g.add(CreateVideo, {
    "images": unsafeRef(rtxVideoSuperResolution_1.id, 0),
    "fps": unsafeRef(getVideoComponents_7.id, 2) as never,
    "audio": unsafeRef(getVideoComponents_7.id, 1),
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
  name: "comfy-workflow-h3-rtx-vsr",
  title: "NVIDIA RTX Video Super-Resolution",
  description: "LoadVideo → RTXVideoSuperResolution → SaveVideo. Gemmy `video h3 upscale --backend rtx`.",
  outputs: [
  {
    "name": "output-0",
    "type": "IMAGE"
  }
],
  models: [],
  environment: "h3",
  nodePacks: [{"id":"comfyui_nvidia_rtx_nodes","name":"ComfyUI_NVIDIA_RTX_Nodes","version":"^0.1.3","repository":"https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI","provides":["RTXVideoSuperResolution"],"source":"registry"}],
};
