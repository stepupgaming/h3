/** T2VA 4-step res_multistep without native SLA. Matched Jev control. */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-t2va/ir.build.ts";

export const buildTemplate = () => buildStage("nosla");
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-t2va-nosla",
  title: "H3 T2VA 4-step res_multistep (no SLA)",
  description:
    "Matched 009jev sampler without H3JevNativeSLAPatch. Gemmy `video h3 generate --mode t2va --no-sla`.",
  nodePacks: ["gemmy-h3-context"],
};
