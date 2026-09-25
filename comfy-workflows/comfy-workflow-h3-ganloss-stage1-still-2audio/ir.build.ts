/** Still-only Eros stage 1 with two live audio refs. */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-ganloss-stage1/ir.build.ts";

export const buildTemplate = () => buildStage(false, false, false, 2);
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-ganloss-stage1-still-2audio",
  title: "Eros still-only stage 1 + two audio refs",
  description:
    "Reference still plus two LoadAudio sockets into Eros SplitSigmas@4. Gemmy `video h3 generate --mode ref2va --ref-image --ref-audio` ×2.",
};
