# comfy-workflow-h3-ref2va-2audio

Same stock Ref2VA graph as `comfy-workflow-h3-ref2va` plus
`ref_audios.ref_audio_0` and `ref_audios.ref_audio_1`.

`gemmy video h3 generate --mode ref2va --ref-image still.png --ref-audio a.wav --ref-audio b.wav --weights <stock Ref2VA>`

This is a new generation with two sound references, not `audio-h3 convert`.
Does not claim a blended speaker.

Default Eros two-stage uses `ganloss-stage1-still-2audio` instead.

AUTHORING SOURCE: `../comfy-workflow-h3-ref2va/ir.build.ts` (`buildTemplate(false, false, 2)`).
Rebuild: `pnpm comfy:build`.
