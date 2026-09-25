/** Stock Ref2VA (single still), 4-step res_multistep, no native SLA. */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-ref2va/ir.build.ts";

export const buildTemplate = () => buildStage(false, false, 0, "nosla");
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-ref2va-nosla",
  title: "H3 Ref2VA 4-step res_multistep (no SLA)",
  description:
    "Matched 009jev sampler without H3JevNativeSLAPatch. Gemmy `video h3 generate --mode ref2va --no-sla`. Skips Eros two-stage.",
  nodePacks: ["gemmy-h3-context"],
};
