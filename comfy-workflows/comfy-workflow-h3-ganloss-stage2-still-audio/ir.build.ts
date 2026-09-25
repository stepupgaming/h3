/** Still-only Eros stage 2 with one live audio ref. */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-ganloss-stage2/ir.build.ts";

export const buildTemplate = () => buildStage(false, false, false, 1);
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-ganloss-stage2-still-audio",
  title: "Eros still-only stage 2 + one audio ref",
  description:
    "Refine the Eros working AV latent with the same still plus LoadAudio. Gemmy `video h3 generate --mode ref2va --ref-image --ref-audio`.",
};
