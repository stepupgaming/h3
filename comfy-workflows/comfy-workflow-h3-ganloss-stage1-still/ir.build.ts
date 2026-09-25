/** Still-only Eros stage 1. Shares the sampler with the crop-video variant. */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-ganloss-stage1/ir.build.ts";

export const buildTemplate = () => buildStage(false);
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-ganloss-stage1-still",
  title: "Eros still-only stage 1",
  description: "Reference still to Eros SplitSigmas@4 working AV latent and preview.",
};
