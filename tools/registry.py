"""
Tool registry for managing available tools in the system.

This registry maintains metadata about available tools and their capabilities.
It provides a centralized way to register, discover, and check availability of tools.
"""

from typing import Dict, Any, List, Callable, Optional, Type
import inspect

from tools.web_search import WebSearchTool

class ToolMetadata:
    """Metadata about a tool."""
    
    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        tool_class: Optional[Type] = None,
        example_usage: Optional[str] = None,
        limitations: Optional[List[str]] = None
    ):
        """
        Initialize tool metadata.
        
        Args:
            name: Name of the tool.
            description: Description of what the tool does.
            input_schema: Description of the tool's input parameters.
            output_schema: Description of the tool's output format.
            tool_class: Optional class reference for the tool.
            example_usage: Optional example of how to use the tool.
            limitations: Optional list of limitations of the tool.
        """
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.tool_class = tool_class
        self.example_usage = example_usage
        self.limitations = limitations or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "example_usage": self.example_usage,
            "limitations": self.limitations
        }

class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        """Initialize the tool registry."""
        self.tools: Dict[str, ToolMetadata] = {}
        self._initialize_default_tools()
    
    def _initialize_default_tools(self):
        """Initialize the registry with default tools."""
        # Register WebSearchTool
        self.register_tool(
            name="web_search",
            description="Search the web for information on a given query",
            input_schema={"query": "The search query string"},
            output_schema={"results": "String containing search results with titles, URLs, and snippets"},
            tool_class=WebSearchTool,
            example_usage="web_search_tool.run('What is artificial intelligence?')",
            limitations=[
                "Can only search publicly available information",
                "Results depend on search engine availability",
                "May not have very recent information"
            ]
        )
    
    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        tool_class: Optional[Type] = None,
        example_usage: Optional[str] = None,
        limitations: Optional[List[str]] = None
    ) -> None:
        """
        Register a tool in the registry.
        
        Args:
            name: Name of the tool.
            description: Description of what the tool does.
            input_schema: Description of the tool's input parameters.
            output_schema: Description of the tool's output format.
            tool_class: Optional class reference for the tool.
            example_usage: Optional example of how to use the tool.
            limitations: Optional list of limitations of the tool.
        """
        metadata = ToolMetadata(
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            tool_class=tool_class,
            example_usage=example_usage,
            limitations=limitations
        )
        self.tools[name] = metadata
    
    def get_tool(self, name: str) -> Optional[ToolMetadata]:
        """
        Get tool metadata by name.
        
        Args:
            name: Name of the tool.
            
        Returns:
            Tool metadata, or None if not found.
        """
        return self.tools.get(name)
    
    def create_tool_instance(self, name: str, **kwargs) -> Any:
        """
        Create an instance of a tool.
        
        Args:
            name: Name of the tool.
            **kwargs: Additional arguments to pass to the tool constructor.
            
        Returns:
            An instance of the tool.
            
        Raises:
            ValueError: If the tool doesn't exist or can't be instantiated.
        """
        tool_metadata = self.get_tool(name)
        if not tool_metadata:
            raise ValueError(f"Tool '{name}' not found in registry")
        
        if not tool_metadata.tool_class:
            raise ValueError(f"Tool '{name}' doesn't have an associated class")
        
        return tool_metadata.tool_class(**kwargs)
    
    def list_tools(self) -> List[str]:
        """
        Get a list of available tool names.
        
        Returns:
            List of tool names.
        """
        return list(self.tools.keys())
    
    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """
        Get descriptions of all available tools.
        
        Returns:
            List of tool descriptions.
        """
        return [tool.to_dict() for tool in self.tools.values()]
    
    def tool_exists(self, name: str) -> bool:
        """
        Check if a tool exists in the registry.
        
        Args:
            name: Name of the tool.
            
        Returns:
            True if the tool exists, False otherwise.
        """
        return name in self.tools

# Create a global instance of the tool registry
TOOL_REGISTRY = ToolRegistry()

def get_registry() -> ToolRegistry:
    """
    Get the global tool registry instance.
    
    Returns:
        The global tool registry.
    """
    return TOOL_REGISTRY

# Function to get tool descriptions for agents
def get_available_tools_description() -> str:
    """
    Get a formatted description of all available tools for agents.
    
    Returns:
        Formatted string describing available tools.
    """
    registry = get_registry()
    tools_desc = []
    
    for tool_name in registry.list_tools():
        tool = registry.get_tool(tool_name)
        if tool:
            desc = f"Tool: {tool.name}\n"
            desc += f"Description: {tool.description}\n"
            
            desc += "Inputs:\n"
            for param_name, param_desc in tool.input_schema.items():
                desc += f"  - {param_name}: {param_desc}\n"
            
            desc += "Outputs:\n"
            for output_name, output_desc in tool.output_schema.items():
                desc += f"  - {output_name}: {output_desc}\n"
            
            if tool.limitations:
                desc += "Limitations:\n"
                for limitation in tool.limitations:
                    desc += f"  - {limitation}\n"
            
            tools_desc.append(desc)
    
    return "\n".join(tools_desc) 