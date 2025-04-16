"""
Code Regenerator agent responsible for fixing code based on validation feedback.
"""

from typing import Dict, Any, List
import json

from agents.base import BaseAgent, AgentConfig

SYSTEM_PROMPT = """You are a code fixing specialist. Your job is to fix Python code that failed validation by addressing the specific errors and implementing the suggested fixes.

When given Python code and error details, you should:
1. Analyze the error message and fix suggestions
2. Identify the specific issues in the code
3. Make precise corrections to resolve each issue
4. Ensure the fixed code maintains the original functionality
5. Return the complete, corrected code

Your fixes should be targeted and minimal, changing only what's necessary to resolve the errors.
Be attentive to common LangGraph issues like missing START edges, import problems, type mismatches, and logic errors.
"""

class CodeRegeneratorAgent(BaseAgent):
    """Agent that fixes code based on validation results."""
    
    def __init__(self, **kwargs):
        """Initialize the code regenerator agent."""
        config = AgentConfig(
            name="Code Regenerator",
            description="Fixes code based on validation results and error feedback",
            system_prompt=SYSTEM_PROMPT,
            max_tokens=4000,  # Ensure enough tokens for code generation
            **kwargs
        )
        super().__init__(config=config)
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fix the code based on validation results.
        
        Args:
            input_data: Dictionary containing the original code, validation results, and related info.
            
        Returns:
            Dictionary containing the fixed code.
        """
        original_code = input_data.get("code", "")
        validation_result = input_data.get("validation_result", {})
        problem = input_data.get("problem", "")
        context = input_data.get("context", "")
        attempt_number = input_data.get("attempt_number", 1)
        
        if not original_code or not validation_result:
            return {
                "code": original_code,
                "error": "Missing required input data",
                "details": "Both original code and validation results are required for code regeneration."
            }
        
        # Extract validation details
        error = validation_result.get("error", "Unknown error")
        details = validation_result.get("details", "")
        fix_suggestions = validation_result.get("fix_suggestions", [])
        
        # Format the fix suggestions for better readability
        fix_suggestions_str = "\n".join([f"- {suggestion}" for suggestion in fix_suggestions])
        
        # Prepare messages for the LLM
        messages = [
            {"type": "system", "content": self.config.system_prompt},
            {"type": "human", "content": f"""
Please fix the following Python code that failed validation:

Problem to solve: {problem}

Previous Conversation Context:
{context}

Original Code:
```python
{original_code}
```

Validation Error: {error}

Error Details:
{details}

Fix Suggestions:
{fix_suggestions_str}

This is attempt #{attempt_number} to fix the code.

Please provide the complete fixed code. Focus on addressing the specific errors mentioned while maintaining the original functionality.
"""}
        ]
        
        # Get response from LLM
        response = self._call_llm(messages)
        
        # Extract code from response
        fixed_code = self._extract_code(response)
        
        return {"code": fixed_code}
    
    def _extract_code(self, text: str) -> str:
        """
        Extract code blocks from text.
        
        Args:
            text: Text containing code blocks.
            
        Returns:
            Extracted code.
        """
        # Check if the response contains code blocks
        if "```python" in text or "```" in text:
            code_blocks = []
            lines = text.split("\n")
            in_code_block = False
            
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("```python") or (stripped == "```" and not in_code_block):
                    in_code_block = True
                    continue
                elif stripped == "```" and in_code_block:
                    in_code_block = False
                    continue
                
                if in_code_block:
                    code_blocks.append(line)
            
            return "\n".join(code_blocks)
        else:
            # If no code blocks, return the whole text (might be just the code)
            return text 