/** Stock Ref2VA (single still) plus one live audio ref. */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-ref2va/ir.build.ts";

export const buildTemplate = () => buildStage(false, false, 1);
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-ref2va-audio",
  title: "H3 Reference to Video-Audio + one audio ref",
  description:
    "Stock Ref2VA with one live still and one LoadAudio on ref_audios.ref_audio_0. Gemmy `video h3 generate --mode ref2va --ref-image --ref-audio --weights`. Not Eros two-stage.",
};
