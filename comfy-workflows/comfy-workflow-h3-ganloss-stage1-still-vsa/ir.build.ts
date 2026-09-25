/** Still-only Eros stage 1 with Ref2VA VSA gate. */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-ganloss-stage1/ir.build.ts";

export const buildTemplate = () => buildStage(false, true);
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-ganloss-stage1-still-vsa",
  title: "Eros still-only stage 1 + VSA",
  description: "Reference still to Eros SplitSigmas@4 with Ref2VAVSAGatePatch. Opt-in --vsa.",
};
