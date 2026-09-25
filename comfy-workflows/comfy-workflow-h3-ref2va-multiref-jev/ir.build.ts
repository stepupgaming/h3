/** Stock Ref2VA two-still with 009jev native SLA. Opt-in `--jev` only. */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-ref2va-multiref/ir.build.ts";

export const buildTemplate = () => buildStage(false, true);
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-ref2va-multiref-jev",
  title: "H3 Ref2VA multi-still + 009jev native SLA",
  description:
    "Opt-in 009jev: Jev-guided native SLA, 4-step res_multistep. Gemmy `video h3 generate --mode ref2va --jev` with two stills.",
  nodePacks: ["gemmy-h3-context"],
};
