#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -n "${PYTHON:-}" ]]; then
  runtime_python="$PYTHON"
elif [[ -x .venv/bin/python ]]; then
  runtime_python=.venv/bin/python
else
  runtime_python=python3
fi
exec "$runtime_python" -m minimax_voice_api
