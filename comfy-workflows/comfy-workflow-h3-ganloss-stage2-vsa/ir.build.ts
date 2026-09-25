/** Crop-video Eros stage 2 with Ref2VA VSA gate (masked edit). */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-ganloss-stage2/ir.build.ts";

export const buildTemplate = () => buildStage(true, true);
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-ganloss-stage2-vsa",
  title: "Eros stage 2 + VSA",
  description: "Crop-video Eros refine with Ref2VAVSAGatePatch. Opt-in --vsa.",
};
