/** I2V / first-frame FL2VA with 009jev native SLA on stock HQ euler/simple. */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-fl2va/ir.build.ts";

export const buildTemplate = () => buildStage("hq");
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-fl2va-jev-hq",
  title: "H3 first-frame + 009jev native SLA (HQ euler)",
  description:
    "Opt-in 009jev native SLA on stock 15/20/32-step euler/simple I2V. Gemmy `video h3 generate --jev --steps 20`. Not the 4-step res_multistep recipe. Not default generate.",
  nodePacks: ["gemmy-h3-context"],
};
