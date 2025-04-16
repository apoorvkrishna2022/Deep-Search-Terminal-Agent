"""
Tools package for Deep Search Terminal Agent.
"""

from tools.web_search import WebSearchTool
from tools.registry import (
    ToolRegistry,
    get_registry,
    get_available_tools_description,
    TOOL_REGISTRY
)

__all__ = [
    "WebSearchTool",
    "ToolRegistry",
    "get_registry",
    "get_available_tools_description",
    "TOOL_REGISTRY"
] 