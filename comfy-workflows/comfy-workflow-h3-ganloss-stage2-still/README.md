# Eros still-only stage 2

Local Gemmy workflow variant. Authoring reuses the sibling ganloss-stage2 graph with video conditioning omitted at build time. Runtime binds the compiled template without Node.

Required parameters: prompt, ref_image, context_latent. No crop_video is required.

Width and height control both conditioning and the latent upscaler. Aspect locking is disabled so both requested dimensions are honored.

From comfy-workflows/: pnpm comfy:build, then pnpm comfy:check. Do not edit generated JSON.
