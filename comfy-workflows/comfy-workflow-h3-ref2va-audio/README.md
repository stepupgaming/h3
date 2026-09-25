# comfy-workflow-h3-ref2va-audio

Same stock Ref2VA graph as `comfy-workflow-h3-ref2va` plus
`ref_audios.ref_audio_0`.

`gemmy video h3 generate --mode ref2va --ref-image still.png --ref-audio a.wav --weights <stock Ref2VA>`

Default Eros two-stage uses `ganloss-stage1-still-audio` instead.

AUTHORING SOURCE: `../comfy-workflow-h3-ref2va/ir.build.ts` (`buildTemplate(false, false, 1)`).
Rebuild: `pnpm comfy:build`.
