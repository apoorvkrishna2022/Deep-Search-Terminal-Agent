"""
Decomposer agent responsible for breaking down the roadmap into tasks for individual agents.
"""

from typing import Dict, Any, List
import json

from agents.base import BaseAgent, AgentConfig
from tools.registry import get_available_tools_description, get_registry

SYSTEM_PROMPT = """You are a task decomposition specialist. Your job is to break down a complex roadmap into well-defined tasks that can be executed by specialized agents.

When given a roadmap, you should:
1. Analyze each step in the roadmap
2. Break down complex steps into smaller, actionable tasks
3. Identify what type of agent would be best suited for each task
4. Define the inputs and outputs for each task
5. Specify the tools or capabilities needed by each agent
6. Determine how agents should communicate with each other
7. ONLY use tools that are available in the system

Your output should allow for maximum parallelization where possible while maintaining necessary dependencies between tasks.

Output your task breakdown as a JSON object with the following structure:
{
  "agent_types": [
    {
      "type": "agent_type_name",
      "description": "What this type of agent does",
      "capabilities": ["capability1", "capability2"],
      "tools": ["tool_name_1", "tool_name_2"]
    },
    ...
  ],
  "tasks": [
    {
      "id": "unique_task_id",
      "title": "Task title",
      "description": "Detailed description of the task",
      "agent_type": "agent_type_name",
      "inputs": ["input1", "input2"],
      "outputs": ["output1", "output2"],
      "dependencies": ["task_id1", "task_id2"],
      "tools": ["tool_name_1", "tool_name_2"],
      "estimated_complexity": "low|medium|high"
    },
    ...
  ],
  "workflow": {
    "initial_tasks": ["task_id1", "task_id2"],
    "terminal_tasks": ["task_id3", "task_id4"]
  },
  "tool_requests": [
    {
      "name": "requested_tool_name",
      "description": "What this tool would do",
      "reason_needed": "Why this tool is needed",
      "workaround": "How to accomplish the task with existing tools"
    },
    ...
  ]
}

IMPORTANT: For the "tools" fields in "agent_types" and "tasks", ONLY specify tools that are actually available in the system. If you need capabilities not provided by available tools, add an entry to the "tool_requests" section and describe a workaround using existing tools.
"""

class DecomposerAgent(BaseAgent):
    """Agent that breaks down roadmaps into tasks for individual agents."""
    
    def __init__(self, **kwargs):
        """Initialize the decomposer agent."""
        config = AgentConfig(
            name="Task Decomposer",
            description="Breaks down complex roadmaps into manageable tasks for specialized agents",
            system_prompt=SYSTEM_PROMPT,
            **kwargs
        )
        super().__init__(config=config)
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Break down a roadmap into tasks for individual agents.
        
        Args:
            input_data: Dictionary containing the roadmap and optional context.
            
        Returns:
            Dictionary containing the task breakdown.
        """
        roadmap = input_data.get("roadmap", {})
        context = input_data.get("context", "")
        
        # Get available tools description
        available_tools = get_available_tools_description()
        
        # Convert roadmap to a string for the LLM
        roadmap_str = json.dumps(roadmap, indent=2)
        
        # Prepare messages for the LLM
        messages = [
            {"type": "system", "content": self.config.system_prompt},
            {"type": "human", "content": f"""
Please break down the following roadmap into tasks for specialized agents:

{roadmap_str}

{context}

Available Tools:
{available_tools}

Consider what types of agents would be needed, what tools they would require, and how they should communicate with each other.
Remember to ONLY use tools that are available in the system. If you need additional tools, add them to the "tool_requests" section.
"""}
        ]
        
        # Get response from LLM
        response = self._call_llm(messages)
        
        # Extract JSON from response
        try:
            # Find JSON in the response if it's embedded in text
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                task_breakdown = json.loads(json_str)
            else:
                task_breakdown = json.loads(response)
                
            # Validate and process any tool requests
            self._process_tool_requests(task_breakdown)
            
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw text
            task_breakdown = {"error": "Failed to parse response", "raw_response": response}
        
        return {"task_breakdown": task_breakdown}
    
    def _process_tool_requests(self, task_breakdown: Dict[str, Any]) -> None:
        """
        Process any tool requests in the task breakdown.
        
        This method logs tool requests for future reference and ensures they're included in the task breakdown.
        
        Args:
            task_breakdown: The task breakdown with possible tool requests.
        """
        if "tool_requests" not in task_breakdown:
            task_breakdown["tool_requests"] = []
        
        # Add comments to each tool request noting if it's already available
        registry = get_registry()
        for tool_request in task_breakdown.get("tool_requests", []):
            if "name" in tool_request:
                tool_name = tool_request["name"]
                if registry.tool_exists(tool_name):
                    tool_request["status"] = "already_available"
                else:
                    tool_request["status"] = "requested" 