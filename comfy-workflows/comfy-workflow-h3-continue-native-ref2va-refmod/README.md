# H3 Native-Guide Continue (Ref2VA still + RefMod)

Continue from a persisted AV latent with one live `--ref-image` and saved `--ref-mod` (Apply, not Text Encode). Do not Apply and Text Encode the same bundle.

AUTHORING SOURCE: `ir.build.ts` (wrapper around `h3-continue-native-ref2va` with `withRefmod=true`). Rebuild: `pnpm comfy:build`.
