/** Still-only Eros refinement. Shares the sampler with the crop-video variant. */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-ganloss-stage2/ir.build.ts";

export const buildTemplate = () => buildStage(false);
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-ganloss-stage2-still",
  title: "Eros still-only stage 2",
  description: "Refine the Eros working AV latent at the requested width and height with the same reference still.",
};
