# 009jev: Jev-guided Native SLA

Experimental opt-in for `gemmy video h3 generate --jev`.

This folder is the **009jev-only** slice of sepiablue-ai/ComfyUI-MiniMax-H3-W4A4-VSA (`exp/jev-adaptive-vsa`): `H3JevNativeSLAPatch` plus `native_sla_worker.py`. It does **not** register W4A4 conversion, `H3V2PreconvertedLoader`, or `H3V2JevAdaptiveVSAPatch`.

- License: GPL-3.0-only (`LICENSE`)
- Method: [009JEV.en.md](009JEV.en.md)
- SDK: dedicated uv env at `runtimes/minimax-h3/jev-sdk` (`typesafe-sdk==0.7.0`), not Comfy's Python
- API key: `TYPESAFE_API_KEY` on the Comfy process (env / Gemmy `env_overrides`). Never in workflows or this tree.

Not a default generate pin. Default `gemmy video h3 generate` must not require this node or the key.
