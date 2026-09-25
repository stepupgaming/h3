/** Still-only Eros stage 1 with one live audio ref. */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-ganloss-stage1/ir.build.ts";

export const buildTemplate = () => buildStage(false, false, false, 1);
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-ganloss-stage1-still-audio",
  title: "Eros still-only stage 1 + one audio ref",
  description:
    "Reference still plus LoadAudio on ref_audios.ref_audio_0 into Eros SplitSigmas@4. Gemmy `video h3 generate --mode ref2va --ref-image --ref-audio`.",
};
