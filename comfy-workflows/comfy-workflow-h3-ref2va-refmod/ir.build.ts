/** Stock Ref2VA (single still) plus saved RefMods via Apply. */
import { buildTemplate as buildStage, manifestMeta as stageMeta } from "../comfy-workflow-h3-ref2va/ir.build.ts";

export const buildTemplate = () => buildStage(false, true);
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-ref2va-refmod",
  title: "H3 Reference to Video-Audio + RefMod",
  description:
    "Live still plus saved RefMods (Apply). Gemmy `video h3 generate --mode ref2va --ref-mod --weights`. Product Eros two-stage uses the ganloss still-refmod packages.",
  nodePacks: ["ComfyUI-MiniMaxH3Mod"],
};
