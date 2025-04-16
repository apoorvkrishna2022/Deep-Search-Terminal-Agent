"""
Planner agent responsible for creating the initial roadmap to solve a problem.
"""

from typing import Dict, Any, List
import json

from agents.base import BaseAgent, AgentConfig
from tools.web_search import WebSearchTool
from tools.registry import get_available_tools_description

SYSTEM_PROMPT = """You are a strategic planner agent. Your job is to create a detailed roadmap for solving a given problem statement.

When given a problem, you should:
1. Break down the problem into logical steps
2. Create a roadmap with clear milestones
3. Define what information is needed at each step
4. Consider multiple approaches and pick the most effective one
5. Structure your output in a clear, organized way
6. Consider any relevant context from previous conversations when planning
7. Use only the tools that are available in the system

Your roadmap should be comprehensive yet flexible, allowing for adjustments as new information comes in.

Output your roadmap as a JSON object with the following structure:
{
  "problem_analysis": "A brief analysis of the problem, identifying key aspects and challenges",
  "roadmap": [
    {
      "step": 1,
      "title": "Step title",
      "description": "Detailed description of what this step accomplishes",
      "required_info": ["Info item 1", "Info item 2"],
      "expected_outcome": "What this step should produce",
      "tools_needed": ["tool_name_1", "tool_name_2"]
    },
    ...
  ],
  "alternative_approaches": [
    {
      "approach": "Alternative approach name",
      "pros": ["Pro 1", "Pro 2"],
      "cons": ["Con 1", "Con 2"],
      "required_tools": ["tool_name_1", "tool_name_2"]
    },
    ...
  ],
  "rationale": "Explanation of why this roadmap is appropriate for the problem"
}

IMPORTANT: For the "tools_needed" and "required_tools" fields, ONLY specify tools that are actually available in the system. If you need capabilities not provided by available tools, think about how to accomplish the task with the tools that are available, or suggest a workaround.
"""

class PlannerAgent(BaseAgent):
    """Agent that creates a strategic roadmap for solving problems."""
    
    def __init__(self, **kwargs):
        """Initialize the planner agent."""
        config = AgentConfig(
            name="Strategic Planner",
            description="Creates detailed roadmaps for solving complex problems",
            system_prompt=SYSTEM_PROMPT,
            tools=[WebSearchTool()],
            **kwargs
        )
        super().__init__(config=config)
        
        # Register tools
        self.web_search = WebSearchTool()
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a roadmap for solving the given problem.
        
        Args:
            input_data: Dictionary containing the problem statement and optional context.
            
        Returns:
            Dictionary containing the roadmap.
        """
        problem = input_data.get("problem", "")
        context = input_data.get("context", "")
        
        # Get available tools description
        available_tools = get_available_tools_description()
        
        # Get relevant information from the web if needed
        search_results = self.web_search.run(problem)
        
        # Prepare messages for the LLM
        messages = [
            {"type": "system", "content": self.config.system_prompt},
            {"type": "human", "content": f"""
Problem Statement: {problem}

{context}

Available Tools:
{available_tools}

Here is some additional context from web searches that might help:
{search_results}

Please create a comprehensive roadmap for solving this problem, using ONLY the tools listed above.
If you need capabilities not provided by these tools, suggest workarounds using the available tools.
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
                roadmap = json.loads(json_str)
            else:
                roadmap = json.loads(response)
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw text
            roadmap = {"error": "Failed to parse response", "raw_response": response}
        
        return {"roadmap": roadmap} 