/** Still-only Eros stage 2 with Ref2VA VSA gate. */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-ganloss-stage2/ir.build.ts";

export const buildTemplate = () => buildStage(false, true);
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-ganloss-stage2-still-vsa",
  title: "Eros still-only stage 2 + VSA",
  description: "Refine the Eros working AV latent with Ref2VAVSAGatePatch. Opt-in --vsa.",
};
