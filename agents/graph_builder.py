"""
Graph Builder agent responsible for creating LangGraph code from the task breakdown.
"""

from typing import Dict, Any, List
import json

from agents.base import BaseAgent, AgentConfig
from tools.registry import get_available_tools_description, get_registry

SYSTEM_PROMPT = """You are a code generation specialist focused on LangGraph. Your job is to create executable Python code that implements a LangGraph workflow based on a task breakdown.

When given a task breakdown, you should:
1. Analyze the agent types and tasks
2. Design a LangGraph structure that represents the workflow
3. Write Python code that implements each agent with their required tools
4. Create the LangGraph connections based on task dependencies
5. Ensure the code is fully functional and executable
6. Include appropriate error handling and logging
7. Support conversation context from previous interactions
8. ONLY use tools that are actually available in the system

Your output should be complete, well-structured Python code that can be executed to solve the original problem.

The code should follow these guidelines:
- Use the correct import for StateGraph: `from langgraph.graph import StateGraph`
- Implement each agent type with appropriate methods
- Include all necessary imports
- Ensure proper tool integration for each agent
- Handle data passing between agents correctly
- Include appropriate type hints
- Follow best practices for Python code style
- Incorporate conversation context from previous interactions when relevant
- ONLY use tools that are available in the system
- **CRITICAL**: Always add at least one edge from START to another node when building the graph
- Make sure to call `workflow.add_edge(START, first_node_name)` to define the entry point

For any tool requests that couldn't be fulfilled, implement a workaround using available tools.

The main function should take a problem statement as input and optionally a context parameter (for previous conversation context), and return the final solution.

Here's a basic template for the LangGraph structure:

```python
import os
from typing import Dict, Any, List, TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import AzureChatOpenAI
from tools.web_search import WebSearchTool  # Import from tools directory, not langchain_openai
# Other imports as needed

# Define state schema
class AgentState(TypedDict):
    # State fields here
    input: str
    context: Optional[str]
    # Add more fields as needed

# Create agents
def create_agent1(state):
    # Agent logic
    # Example of using WebSearchTool from tools directory
    search_tool = WebSearchTool()
    search_results = search_tool.run("your search query")
    return {"output": search_results}

# Define state transitions
def should_continue(state):
    # Transition logic
    return "next_node" if condition else END

# Create main function
def solve(problem: str, context: str = "") -> Dict[str, Any]:
    # Setup LLM and tools
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-03-15-preview"),
    )
    
    # Build the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent1", create_agent1)
    
    # CRITICAL: Add edges, always including at least one from START
    workflow.add_edge(START, "agent1")
    workflow.add_edge("agent1", END)
    
    # Compile
    app = workflow.compile()
    
    # Run the graph
    result = app.invoke({"input": problem, "context": context})
    
    # Return final output instead of the full state
    # Extract the most relevant information as the final result
    if "output" in result:
        return result["output"]
    else:
        # Look for other relevant fields in the state
        final_output = {}
        for key in result:
            if key not in ["input", "context"] and result[key] is not None:
                final_output[key] = result[key]
        return final_output
```
"""

class GraphBuilderAgent(BaseAgent):
    """Agent that creates LangGraph code from a task breakdown."""
    
    def __init__(self, **kwargs):
        """Initialize the graph builder agent."""
        config = AgentConfig(
            name="Graph Builder",
            description="Creates executable LangGraph code from task breakdowns",
            system_prompt=SYSTEM_PROMPT,
            max_tokens=4000,  # Ensure enough tokens for code generation
            **kwargs
        )
        super().__init__(config=config)
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create LangGraph code from a task breakdown.
        
        Args:
            input_data: Dictionary containing the task breakdown, original problem, and optional context.
            
        Returns:
            Dictionary containing the generated code.
        """
        task_breakdown = input_data.get("task_breakdown", {})
        problem = input_data.get("problem", "")
        roadmap = input_data.get("roadmap", {})
        context = input_data.get("context", "")
        
        # Get available tools
        available_tools = get_available_tools_description()
        
        # Convert data to strings for the LLM
        task_breakdown_str = json.dumps(task_breakdown, indent=2)
        roadmap_str = json.dumps(roadmap, indent=2) if roadmap else ""
        
        # Extract tool requests if any
        tool_requests = task_breakdown.get("tool_requests", [])
        tool_requests_str = json.dumps(tool_requests, indent=2) if tool_requests else "No additional tool requests."
        
        # Prepare messages for the LLM
        messages = [
            {"type": "system", "content": self.config.system_prompt},
            {"type": "human", "content": f"""
Please generate executable Python code that implements a LangGraph workflow for solving the following problem:

Problem: {problem}

Previous Conversation Context:
{context}

Roadmap:
{roadmap_str}

Task Breakdown:
{task_breakdown_str}

Available Tools:
{available_tools}

Tool Requests and Workarounds:
{tool_requests_str}

The code should create a LangGraph structure that connects all the agents according to the workflow defined in the task breakdown.
Each agent should be implemented with appropriate methods and tools.
The code should be complete and executable, including all necessary imports.

Important: 
1. The main or solve function should accept two parameters:
   - problem (str): The current problem to solve
   - context (str, optional): Previous conversation context that might be relevant

2. Make sure to use the correct import for StateGraph: `from langgraph.graph import StateGraph`

3. ONLY use tools that are available in the system. For any requested tools that are not available, implement the suggested workarounds.

4. CRITICAL: Always add at least one edge from START to another node in your graph:
   ```python
   from langgraph.graph import START
   workflow.add_edge(START, "first_node_name")
   ```

If no context is available, the function should still work with just the problem parameter.
"""}
        ]
        
        # Get response from LLM
        response = self._call_llm(messages)
        
        # Extract code from response
        code = self._extract_code(response)
        
        return {"code": code}
    
    def _extract_code(self, text: str) -> str:
        """
        Extract code blocks from text.
        
        Args:
            text: Text containing code blocks.
            
        Returns:
            Extracted code.
        """
        # Check if the response contains code blocks
        if "```python" in text:
            code_blocks = []
            lines = text.split("\n")
            in_code_block = False
            
            for line in lines:
                if line.strip().startswith("```python"):
                    in_code_block = True
                    continue
                elif line.strip() == "```" and in_code_block:
                    in_code_block = False
                    continue
                
                if in_code_block:
                    code_blocks.append(line)
            
            return "\n".join(code_blocks)
        else:
            # If no code blocks, return the whole text
            return text 