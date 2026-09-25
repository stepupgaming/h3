"""Bind a compiled Comfy prompt template.

Build time (Node, official compiler):

    workflow.ir.json  →  @stepupgaming/comfy-workflows compile  →  prompt.template.json

Runtime (Python, no Node, no IR translation):

    prompt.template.json + params  →  Comfy API/prompt JSON

This module does **not** compile Graph IR. It only substitutes ``{"$param": name}``
placeholders that the official compiler left in the generated template, plus
``{"$int": "..."}`` tags if any remain. Connections, bypass lowering, node
ordering, and serialization come from the compiler artifact.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PACKAGES_ROOT = Path(__file__).resolve().parent


class MaterializeError(RuntimeError):
    pass


def packages_root() -> Path:
    override = os.environ.get("GEMMY_COMFY_WORKFLOWS")
    if override:
        return Path(override).expanduser().resolve()
    return PACKAGES_ROOT


def package_dir(name: str) -> Path:
    root = packages_root()
    direct = root / name
    if (direct / "package.json").is_file():
        return direct
    for child in root.iterdir():
        if not child.is_dir():
            continue
        pj = child / "package.json"
        if not pj.is_file():
            continue
        try:
            meta = json.loads(pj.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if meta.get("name") == name:
            return child
    raise MaterializeError(f"workflow package not found: {name} (root={root})")


def load_manifest(pkg: Path) -> dict[str, Any]:
    man_path = pkg / "comfy.workflow.json"
    if not man_path.is_file():
        raise MaterializeError(f"missing comfy.workflow.json in {pkg}")
    return json.loads(man_path.read_text(encoding="utf-8"))


def load_ir(pkg: Path) -> dict[str, Any]:
    """Authoring IR — tests/docs only. Runtime never compiles this."""
    ir_path = pkg / "workflow.ir.json"
    if not ir_path.is_file():
        raise MaterializeError(f"missing workflow.ir.json in {pkg}")
    return json.loads(ir_path.read_text(encoding="utf-8"))


def load_prompt_template(pkg: Path) -> Any:
    path = pkg / "prompt.template.json"
    if not path.is_file():
        raise MaterializeError(
            f"missing prompt.template.json in {pkg} "
            "(run `npm run build` in comfy-workflows/ to compile IR with the official compiler)"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _is_param_ref(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value.keys()) == {"$param"}
        and isinstance(value.get("$param"), str)
    )


def _untag_int(value: Any) -> Any:
    if isinstance(value, dict) and set(value.keys()) == {"$int"}:
        raw = value["$int"]
        try:
            return int(raw)
        except (TypeError, ValueError):
            return raw
    return value


def _effective_params(manifest: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    bound: dict[str, Any] = {}
    for name, spec in (manifest.get("parameters") or {}).items():
        if isinstance(spec, dict) and "default" in spec:
            bound[name] = spec["default"]
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                bound[k] = v
    for name, spec in (manifest.get("parameters") or {}).items():
        if not isinstance(spec, dict):
            continue
        if spec.get("required") and name not in bound:
            raise MaterializeError(f"required template param {name!r} was not bound")
    return bound


def bind_prompt(template: Any, bound: dict[str, Any]) -> Any:
    """Walk a compiled prompt template and substitute {$param} leaves."""
    if _is_param_ref(template):
        name = template["$param"]
        if name not in bound:
            raise MaterializeError(f"unbound template param {name!r}")
        return _untag_int(bound[name])
    tagged = _untag_int(template)
    if tagged is not template:
        return tagged
    if isinstance(template, list):
        return [bind_prompt(v, bound) for v in template]
    if isinstance(template, dict):
        return {k: bind_prompt(v, bound) for k, v in template.items()}
    return template


def materialize_package(name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    pkg = package_dir(name)
    manifest = load_manifest(pkg)
    template = load_prompt_template(pkg)
    bound = _effective_params(manifest, params)
    graph = bind_prompt(template, bound)
    if not isinstance(graph, dict) or not graph:
        raise MaterializeError(f"compiled prompt for {name} is empty")
    blob = json.dumps(graph)
    if '"$param"' in blob:
        raise MaterializeError(f"unbound $param remained in {name}")
    return graph


def iter_packages() -> list[Path]:
    root = packages_root()
    out: list[Path] = []
    for child in sorted(root.iterdir()):
        if (
            child.is_dir()
            and (child / "workflow.ir.json").is_file()
            and (child / "package.json").is_file()
        ):
            out.append(child)
    return out
