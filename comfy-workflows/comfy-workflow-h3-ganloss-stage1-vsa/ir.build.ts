/** Crop-video Eros stage 1 with Ref2VA VSA gate (masked edit). */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-ganloss-stage1/ir.build.ts";

export const buildTemplate = () => buildStage(true, true);
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-ganloss-stage1-vsa",
  title: "Eros stage 1 + VSA",
  description: "Crop-video Eros SplitSigmas@4 with Ref2VAVSAGatePatch. Opt-in --vsa.",
};
