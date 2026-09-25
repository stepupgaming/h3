"""Tensor-tree pack for MiniMax conditioning. No Comfy import."""

from __future__ import annotations

import torch


def pack_tree(obj):
    """Return (safetensors dict, JSON tree). Tensors are replaced with ``{"$t": key}``."""
    blobs: dict[str, torch.Tensor] = {}

    def walk(value, path: str):
        if isinstance(value, torch.Tensor):
            key = f"t{len(blobs)}"
            blobs[key] = value.detach().cpu().contiguous()
            return {"$t": key}
        if getattr(value, "is_nested", False):
            return {"$nested": [walk(part, f"{path}[{i}]") for i, part in enumerate(value.tensors)]}
        if isinstance(value, dict):
            return {str(k): walk(v, f"{path}.{k}") for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [walk(v, f"{path}[{i}]") for i, v in enumerate(value)]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(
            f"cannot store conditioning at {path}: {type(value).__module__}.{type(value).__name__}"
        )

    return blobs, walk(obj, "$")


def unpack_tree(blobs: dict, tree):
    """Inverse of pack_tree. Nested markers stay as ``{"$nested": [...]}`` for the node to wrap."""
    if isinstance(tree, dict) and set(tree) == {"$t"}:
        key = tree["$t"]
        if key not in blobs:
            raise KeyError(f"conditioning tensor {key} missing")
        return blobs[key]
    if isinstance(tree, dict) and set(tree) == {"$nested"}:
        return {"$nested": [unpack_tree(blobs, part) for part in tree["$nested"]]}
    if isinstance(tree, dict):
        return {k: unpack_tree(blobs, v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [unpack_tree(blobs, v) for v in tree]
    return tree
