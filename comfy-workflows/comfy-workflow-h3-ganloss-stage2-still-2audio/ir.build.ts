/** Still-only Eros stage 2 with two live audio refs. */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-ganloss-stage2/ir.build.ts";

export const buildTemplate = () => buildStage(false, false, false, 2);
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-ganloss-stage2-still-2audio",
  title: "Eros still-only stage 2 + two audio refs",
  description:
    "Refine the Eros working AV latent with the same still plus two LoadAudio sockets.",
};
