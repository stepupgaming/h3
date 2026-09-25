/** I2V / first-frame FL2VA, 4-step res_multistep, no native SLA patch. Matched Jev control. */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-fl2va/ir.build.ts";

export const buildTemplate = () => buildStage("nosla");
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-fl2va-nosla",
  title: "H3 first-frame 4-step res_multistep (no SLA)",
  description:
    "Matched 009jev sampler without H3JevNativeSLAPatch. Gemmy `video h3 generate --no-sla` (I2V). Not the default 20-step euler path.",
  nodePacks: ["gemmy-h3-context"],
};
