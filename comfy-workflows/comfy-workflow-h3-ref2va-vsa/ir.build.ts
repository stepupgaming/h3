/** Stock Ref2VA (single still) with Ref2VAVSAGatePatch. */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-ref2va/ir.build.ts";

export const buildTemplate = () => buildStage(true);
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-ref2va-vsa",
  title: "H3 Reference to Video-Audio + VSA",
  description: "Stock Comfy-Org Ref2VA with Ref2VAVSAGatePatch. Gemmy `video h3 generate --mode ref2va --vsa --weights`.",
};
