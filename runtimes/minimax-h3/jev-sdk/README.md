# H3 009jev SDK interpreter

Isolated uv environment for `typesafe-sdk==0.7.0`. `H3JevNativeSLAPatch` runs `native_sla_worker.py` with this Python (`sdk_python`). Do not install the SDK into the H3 Comfy venv.

```
uv sync
```

Gemmy `--jev` fail-closes if `.venv/Scripts/python.exe` is missing.
