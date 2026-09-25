/** T2VA with 009jev native SLA (4-step res_multistep). Opt-in `--jev` only. */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-t2va/ir.build.ts";

export const buildTemplate = () => buildStage(true);
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-t2va-jev",
  title: "H3 T2VA + 009jev native SLA",
  description:
    "Opt-in 009jev: Jev-guided native SLA, 4-step res_multistep. Gemmy `video h3 generate --mode t2va --jev`. Not the default generate path.",
  nodePacks: ["gemmy-h3-context"],
};
