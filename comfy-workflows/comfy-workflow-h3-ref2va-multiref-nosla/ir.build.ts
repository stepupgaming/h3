/** Stock Ref2VA two-still, 4-step res_multistep, no native SLA. */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-ref2va-multiref/ir.build.ts";

export const buildTemplate = () => buildStage(false, "nosla");
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-ref2va-multiref-nosla",
  title: "H3 Ref2VA multi-still 4-step res_multistep (no SLA)",
  description:
    "Matched 009jev sampler without H3JevNativeSLAPatch. Gemmy `video h3 generate --mode ref2va --no-sla` with two stills.",
  nodePacks: ["gemmy-h3-context"],
};
