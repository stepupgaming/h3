/** Stock Ref2VA two-still with 009jev native SLA on HQ euler/simple. */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-ref2va-multiref/ir.build.ts";

export const buildTemplate = () => buildStage(false, "hq");
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-ref2va-multiref-jev-hq",
  title: "H3 Ref2VA multi-still + 009jev native SLA (HQ euler)",
  description:
    "Opt-in 009jev native SLA on stock 15/20/32-step euler/simple two-still Ref2VA. Not Eros two-stage.",
  nodePacks: ["gemmy-h3-context"],
};
