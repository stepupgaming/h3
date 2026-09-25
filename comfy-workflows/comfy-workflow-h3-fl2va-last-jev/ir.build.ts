/** First+last FL2VA with 009jev native SLA on stock HQ euler/simple. */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-fl2va-last/ir.build.ts";

export const buildTemplate = () => buildStage(true);
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-fl2va-last-jev",
  title: "H3 first+last + 009jev native SLA (HQ euler)",
  description:
    "Opt-in 009jev native SLA on first+last FL2VA at stock euler/simple. Gemmy `video h3 generate --mode fl2va --jev --steps 20`. No 4-step res_multistep variant.",
  nodePacks: ["gemmy-h3-context"],
};
