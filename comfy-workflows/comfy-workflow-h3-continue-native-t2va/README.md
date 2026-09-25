# H3 Native-Guide Continue (T2VA)

Continue an H3 T2VA clip from a persisted AV latent. The previous tail is attached as native `minimax_keyframes` (no VAE round trip, no denoise mask). Every delivered frame is generated.

AUTHORING SOURCE: `ir.build.ts`.
GENERATED: `workflow.ir.json`, `comfy.workflow.json`, `prompt.template.json` — do not hand-edit.
Rebuild with `pnpm comfy:build` from `comfy-workflows/`.

## Required parameters

- `prompt`
- `context_latent`

`GemmyH3LatentTailGuide` is `rawNode` until `environments/h3` is recaptured.
