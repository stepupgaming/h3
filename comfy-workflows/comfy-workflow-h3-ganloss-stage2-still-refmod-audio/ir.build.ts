/** Still-only Eros stage 2 with RefMods and one live audio ref. */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-ganloss-stage2/ir.build.ts";

export const buildTemplate = () => buildStage(false, false, true, 1);
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-ganloss-stage2-still-refmod-audio",
  title: "Eros still-only stage 2 + RefMod + one audio ref",
  description:
    "Refine the Eros working AV latent with the same still, saved RefMods, and LoadAudio.",
  nodePacks: ["ComfyUI-MiniMaxH3Mod"],
};
