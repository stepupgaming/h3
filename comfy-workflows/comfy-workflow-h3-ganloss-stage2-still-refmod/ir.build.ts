/** Still-only Eros refinement with saved RefMods (Apply). */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-ganloss-stage2/ir.build.ts";

export const buildTemplate = () => buildStage(false, false, true);
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-ganloss-stage2-still-refmod",
  title: "Eros still-only stage 2 + RefMod",
  description:
    "Refine the Eros working AV latent with the same still plus saved RefMods. Gemmy `video h3 generate --mode ref2va --ref-mod`.",
  nodePacks: ["ComfyUI-MiniMaxH3Mod"],
};
