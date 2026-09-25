/**
 * AUTHORING SOURCE. Do not hand-edit generated workflow.ir.json / comfy.workflow.json.
 *
 * Target environment: h3
 * Generated node SDK: environments/h3/nodes
 *
 * Rebuild: pnpm comfy:build
 *
 * PixelForge classes are absent from the last environments/h3 snapshot, so they
 * stay rawNode until recapture + codegen.
 */
import { workflow } from "@stepupgaming/comfy-workflows";
import { unsafeRef } from "@stepupgaming/comfy-workflows";
import type { Graph } from "@stepupgaming/comfy-workflows";
import {
  LoadVideo,
  SaveAnimatedWEBP,
  SaveImage,
} from "../environments/h3/nodes/registry.ts";

export function buildTemplate(): Graph {
  const g = workflow("comfy-workflow-h3-pixelforge");
  const video = g.param("video", { type: "combo" });
  const every_nth = g.param("every_nth", { type: "int", default: 2 });
  const start_offset = g.param("start_offset", { type: "int", default: 0 });
  const max_frames = g.param("max_frames", { type: "int", default: 0 });
  const key_color = g.param("key_color", { type: "string", default: "auto" });
  const loop_mode = g.param("loop_mode", { type: "string", default: "off" });
  const filename_prefix = g.param("filename_prefix", { type: "string", default: "gemmy/pixelforge/sprite" });
  const sheet_prefix = g.param("sheet_prefix", { type: "string", default: "gemmy/pixelforge/sheet" });
  const gif_prefix = g.param("gif_prefix", { type: "string", default: "gemmy/pixelforge/gif" });
  const webp_prefix = g.param("webp_prefix", { type: "string", default: "gemmy/pixelforge/preview" });
  const tag_name = g.param("tag_name", { type: "string", default: "run" });
  const fps = g.param("fps", { type: "float", default: 12 });

  const loadVideo = g.add(LoadVideo, {
    file: video,
  }, { id: "1" });

  const frames = g.rawNode(
    "PixelForgeVideoToFrames",
    {
      video: unsafeRef(loadVideo.id, 0),
      every_nth,
      start_offset,
      max_frames,
    },
    {
      outputs: [
        { name: "images", type: "IMAGE" },
        { name: "fps_effective", type: "FLOAT" },
        { name: "info", type: "STRING" },
      ],
      id: "2",
    },
  );

  const keyed = g.rawNode(
    "PixelForgeChromaKey",
    {
      images: unsafeRef(frames.id, 0),
      key_color,
      tolerance: 0.25,
      softness: 0.0,
      despill: true,
      method: "flood",
      shadow_tolerance: 1.0,
      key_interior: true,
      interior_tolerance: 0.5,
      matte_erode: 1,
      subject_rescue: true,
      interior_max_area: 2.0,
      temporal_alpha: true,
      drop_detached: 5.0,
    },
    {
      outputs: [
        { name: "images", type: "IMAGE" },
        { name: "alpha", type: "MASK" },
      ],
      id: "3",
    },
  );

  const crop = g.rawNode(
    "PixelForgeAutoCrop",
    {
      images: unsafeRef(keyed.id, 0),
      bbox_mode: "union",
      anchor: "bottom_center",
      padding: 8,
      size_multiple: 8,
      out_size: 0,
      canvas_mode: "content",
      canvas_width: 200,
      canvas_height: 200,
      placement: "bottom_center",
      offset_x: 0,
      offset_y: 0,
      alpha: unsafeRef(keyed.id, 1),
    },
    {
      outputs: [
        { name: "images", type: "IMAGE" },
        { name: "alpha", type: "MASK" },
        { name: "crop_info", type: "STRING" },
      ],
      id: "5",
    },
  );

  const looped = g.rawNode(
    "PixelForgeLoopTrim",
    {
      images: unsafeRef(crop.id, 0),
      mode: loop_mode,
      max_loop_error: 0.06,
      search_tail_fraction: 0.5,
      alpha: unsafeRef(crop.id, 1),
    },
    {
      outputs: [
        { name: "images", type: "IMAGE" },
        { name: "alpha", type: "MASK" },
        { name: "report", type: "STRING" },
      ],
      id: "6",
    },
  );

  const dedup = g.rawNode(
    "PixelForgeFrameDedup",
    {
      images: unsafeRef(looped.id, 0),
      threshold: 0.01,
      alpha: unsafeRef(looped.id, 1),
    },
    {
      outputs: [
        { name: "images", type: "IMAGE" },
        { name: "alpha", type: "MASK" },
        { name: "durations_json", type: "STRING" },
      ],
      id: "7",
    },
  );

  const sheet = g.rawNode(
    "PixelForgeSheetPack",
    {
      images: unsafeRef(dedup.id, 0),
      columns: 0,
      padding: 0,
      bg_color: "#000000",
      fps_for_durations: fps,
      alpha: unsafeRef(dedup.id, 1),
      durations_json: unsafeRef(dedup.id, 2),
    },
    {
      outputs: [
        { name: "sheet", type: "IMAGE" },
        { name: "sheet_alpha", type: "MASK" },
        { name: "sheet_json", type: "STRING" },
      ],
      id: "8",
    },
  );

  const exportAse = g.rawNode(
    "PixelForgeAsepriteExport",
    {
      images: unsafeRef(dedup.id, 0),
      filename_prefix,
      tag_name,
      fps,
      build_aseprite: false,
      aseprite_path: "",
      alpha: unsafeRef(dedup.id, 1),
      durations_json: unsafeRef(dedup.id, 2),
    },
    {
      outputs: [{ name: "report", type: "STRING" }],
      id: "9",
    },
  );

  const gif = g.rawNode(
    "PixelForgeSaveGIF",
    {
      images: unsafeRef(dedup.id, 0),
      filename_prefix: gif_prefix,
      fps,
      loop: true,
      upscale: 1,
      preview_webp: true,
      alpha: unsafeRef(dedup.id, 1),
      durations_json: unsafeRef(dedup.id, 2),
    },
    {
      outputs: [{ name: "report", type: "STRING" }],
      id: "10",
    },
  );

  const saveSheet = g.add(SaveImage, {
    images: unsafeRef(sheet.id, 0),
    filename_prefix: sheet_prefix,
  }, { id: "11" });

  const saveWebp = g.add(SaveAnimatedWEBP, {
    images: unsafeRef(dedup.id, 0),
    filename_prefix: webp_prefix,
    fps,
    lossless: true,
    quality: 80,
    method: "default",
  }, { id: "12" });

  g.output(saveSheet.out(0));
  g.output(saveWebp.out(0));
  g.output(exportAse.out(0));
  g.output(gif.out(0));
  return g.toGraph();
}

export default buildTemplate;

export const manifestMeta = {
  name: "comfy-workflow-h3-pixelforge",
  title: "H3 PixelForge sprites",
  description: "MiniMax-H3 clip → PixelForge keyed + cropped sprite loop, sheet, GIF. No full-frame 8-bit crush. Gemmy `video h3 sprites`.",
  outputs: [
    { name: "output-0", type: "IMAGE" },
    { name: "output-1", type: "IMAGE" },
    { name: "output-2", type: "STRING" },
    { name: "output-3", type: "STRING" },
  ],
  models: [],
  environment: "h3",
  nodePacks: ["ComfyUI-PixelForge-H3"],
};
