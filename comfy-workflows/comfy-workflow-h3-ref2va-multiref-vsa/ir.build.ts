/** Stock Ref2VA multi-still with Ref2VAVSAGatePatch. */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-ref2va-multiref/ir.build.ts";

export const buildTemplate = () => buildStage(true);
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-ref2va-multiref-vsa",
  title: "H3 Ref2VA multi-ref + VSA",
  description: "Stock Ref2VA two-still graph with Ref2VAVSAGatePatch. Gemmy `--vsa --weights`.",
};
