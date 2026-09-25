/** Still-only Eros stage 1 with saved RefMods (Apply). */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-ganloss-stage1/ir.build.ts";

export const buildTemplate = () => buildStage(false, false, true);
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-ganloss-stage1-still-refmod",
  title: "Eros still-only stage 1 + RefMod",
  description:
    "Reference still plus saved RefMods into Eros SplitSigmas@4. Gemmy `video h3 generate --mode ref2va --ref-mod`.",
  nodePacks: ["ComfyUI-MiniMaxH3Mod"],
};
