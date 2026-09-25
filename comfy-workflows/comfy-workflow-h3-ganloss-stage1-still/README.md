# Eros still-only stage 1

Local Gemmy workflow variant. Authoring reuses the sibling ganloss-stage1 graph with video conditioning omitted at build time. Runtime binds the compiled template without Node.

Required parameters: prompt, ref_image. No crop_video is required.

SplitSigmas@4 writes the working AV latent for stage 2.

From comfy-workflows/: pnpm comfy:build, then pnpm comfy:check. Do not edit generated JSON.
