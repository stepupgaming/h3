/** Native-guide continue: live still plus saved RefMods via Apply. */
import {
  buildTemplate as buildStage,
  manifestMeta as stageMeta,
} from "../comfy-workflow-h3-continue-native-ref2va/ir.build.ts";

export const buildTemplate = () => buildStage(true);
export default buildTemplate;
export const manifestMeta = {
  ...stageMeta,
  name: "comfy-workflow-h3-continue-native-ref2va-refmod",
  title: "H3 Native-Guide Continue (Ref2VA still + RefMod)",
  description:
    "Native-guide continue with one live --ref-image plus saved RefMods (Apply). Default UNET is Ref2VA.",
  nodePacks: ["gemmy-h3-context", "ComfyUI-MiniMaxH3Mod"],
};
