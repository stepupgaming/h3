"""ComfyUI entry point when this repository is installed as a custom node."""

# Pytest imports a repository-root ``__init__.py`` without a package context.
# ComfyUI always loads custom-node folders as packages.
if __package__:
    from .comfyui_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
