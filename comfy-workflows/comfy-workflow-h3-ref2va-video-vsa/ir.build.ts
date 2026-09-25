/** Stock Ref2VA video-ref with Ref2VAVSAGatePatch. */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-ref2va-video/ir.build.ts";

export const buildTemplate = () => buildStage(true);
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-ref2va-video-vsa",
  title: "H3 Ref2VA video-ref + VSA",
  description: "Stock Ref2VA video-reference graph with Ref2VAVSAGatePatch. Gemmy `--vsa --weights`.",
};
