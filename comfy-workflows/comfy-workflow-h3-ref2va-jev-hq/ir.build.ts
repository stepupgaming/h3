/** Stock Ref2VA (single still) with 009jev native SLA on HQ euler/simple. Not Eros two-stage. */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-ref2va/ir.build.ts";

export const buildTemplate = () => buildStage(false, false, 0, "hq");
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-ref2va-jev-hq",
  title: "H3 Ref2VA + 009jev native SLA (HQ euler)",
  description:
    "Opt-in 009jev native SLA on stock 15/20/32-step euler/simple Ref2VA. Gemmy `video h3 generate --mode ref2va --weights <stock> --jev --steps 20`. Skips Eros two-stage.",
  nodePacks: ["gemmy-h3-context"],
};
