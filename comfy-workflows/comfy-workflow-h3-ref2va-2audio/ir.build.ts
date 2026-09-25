/** Stock Ref2VA (single still) plus two live audio refs. */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-ref2va/ir.build.ts";

export const buildTemplate = () => buildStage(false, false, 2);
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-ref2va-2audio",
  title: "H3 Reference to Video-Audio + two audio refs",
  description:
    "Stock Ref2VA with one live still and two LoadAudio nodes (ref_audio_0 + ref_audio_1). Gemmy `video h3 generate --mode ref2va --ref-image --ref-audio a --ref-audio b --weights`. Not convert. Does not claim a blended speaker.",
};
