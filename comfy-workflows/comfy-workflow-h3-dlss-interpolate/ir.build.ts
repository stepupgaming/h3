/**
 * AUTHORING SOURCE. Do not hand-edit generated workflow.ir.json / comfy.workflow.json.
 *
 * Target environment: h3
 * Generated node SDK: environments/h3/nodes
 *
 * Rebuild: pnpm comfy:build
 *
 * NvidiaDLSSFrameInterpolation is absent from the last environments/h3 snapshot,
 * so it stays rawNode until recapture + codegen.
 */
import { workflow } from "@stepupgaming/comfy-workflows";
import { unsafeRef } from "@stepupgaming/comfy-workflows";
import type { Graph } from "@stepupgaming/comfy-workflows";
import {
  LoadVideo,
  SaveVideo,
} from "../environments/h3/nodes/registry.ts";

export function buildTemplate(): Graph {
  const g = workflow("comfy-workflow-h3-dlss-interpolate");
  const video = g.param("video", { type: "combo" });
  const output_fps = g.param("output_fps", { type: "string", default: "48" });
  const dlss_engine = g.param("dlss_engine", { type: "string", default: "Auto" });
  const encoding_quality = g.param("encoding_quality", { type: "string", default: "Max" });
  const video_codec = g.param("video_codec", { type: "string", default: "H.264" });
  const container = g.param("container", { type: "string", default: "MP4" });
  const filename_prefix = g.param("filename_prefix", { type: "string", default: "gemmy/dlss/interpolate" });

  const loadVideo = g.add(LoadVideo, {
    file: video,
  }, { id: "1" });

  const interpolated = g.rawNode(
    "NvidiaDLSSFrameInterpolation",
    {
      video: unsafeRef(loadVideo.id, 0),
      output_fps,
      dlss_engine,
      encoding_quality,
      video_codec,
      container,
      rename: "Auto",
      custom_suffix: "_DLSSFG",
      hdr_mode: false,
    },
    {
      outputs: [
        { name: "video", type: "VIDEO" },
        { name: "report", type: "STRING" },
      ],
      id: "2",
    },
  );

  const saveVideo = g.add(SaveVideo, {
    filename_prefix,
    format: "mp4",
    codec: "auto" as never,
    video: unsafeRef(interpolated.id, 0),
  }, { id: "3" });

  g.output(saveVideo.out(0));
  g.output(interpolated.out(1));
  return g.toGraph();
}

export default buildTemplate;

export const manifestMeta = {
  name: "comfy-workflow-h3-dlss-interpolate",
  title: "H3 DLSS frame interpolate",
  description: "Post-process a finished MiniMax-H3 MP4 with NVIDIA DLSS Frame Generation. Gemmy `video h3 interpolate`. Not an upscale path; not inside continue/loop encode.",
  outputs: [
    { name: "output-0", type: "VIDEO" },
    { name: "output-1", type: "STRING" },
  ],
  models: [],
  environment: "h3",
  nodePacks: ["ComfyUI-NVIDIA-DLSS-Frame-Interpolation"],
};
