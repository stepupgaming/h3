# SPDX-License-Identifier: GPL-3.0-only
"""009jev only: Jev-guided native SLA. Do not import the W4A4/VSA loader path."""

from .native_sla import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
