/** Stock Ref2VA (single still) with 009jev native SLA. Opt-in `--jev` only; not Eros two-stage. */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-ref2va/ir.build.ts";

export const buildTemplate = () => buildStage(false, false, 0, true);
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-ref2va-jev",
  title: "H3 Ref2VA + 009jev native SLA",
  description:
    "Opt-in 009jev: Jev-guided native SLA, 4-step res_multistep. Gemmy `video h3 generate --mode ref2va --jev`. Skips Eros two-stage. Not the default generate path.",
  nodePacks: ["gemmy-h3-context"],
};
