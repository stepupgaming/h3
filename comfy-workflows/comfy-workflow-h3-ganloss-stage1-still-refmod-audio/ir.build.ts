/** Still-only Eros stage 1 with RefMods and one live audio ref. */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-ganloss-stage1/ir.build.ts";

export const buildTemplate = () => buildStage(false, false, true, 1);
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-ganloss-stage1-still-refmod-audio",
  title: "Eros still-only stage 1 + RefMod + one audio ref",
  description:
    "Reference still, saved RefMods, and LoadAudio into Eros SplitSigmas@4. Gemmy `video h3 generate --mode ref2va --ref-image --ref-mod --ref-audio`.",
  nodePacks: ["ComfyUI-MiniMaxH3Mod"],
};
