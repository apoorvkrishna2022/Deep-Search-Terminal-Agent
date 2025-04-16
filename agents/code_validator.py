"""
Code Validator agent responsible for validating and executing the generated code.
"""

from typing import Dict, Any, List
import tempfile
import os
import sys
import importlib.util
import traceback
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from agents.base import BaseAgent, AgentConfig

SYSTEM_PROMPT = """You are a code validation specialist. Your job is to validate and execute Python code to check if it runs without errors.

When given Python code, you should:
1. Execute the code safely
2. Capture all outputs and errors
3. Provide detailed error information if the code fails
4. Suggest potential fixes for common errors

You are specifically designed to validate LangGraph workflows and report on any issues that need to be fixed.
"""

console = Console()

class CodeValidatorAgent(BaseAgent):
    """
    Agent responsible for validating the generated code to ensure it runs without errors.
    """
    
    def __init__(self, **kwargs):
        """Initialize the code validator agent."""
        config = AgentConfig(
            name="Code Validator",
            description="Validates and executes generated code to ensure it runs without errors",
            system_prompt=SYSTEM_PROMPT,
            **kwargs
        )
        super().__init__(config=config)
    
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the code validator agent.
        
        Args:
            inputs: The inputs for the agent, including the code to validate.
            
        Returns:
            The validation result, including success status, error details, and fix suggestions.
        """
        # Get the generated code
        code = inputs.get("code", "")
        problem = inputs.get("problem", "")
        context = inputs.get("context", "")
        
        if not code:
            return {
                "valid": False,
                "error": "No code provided",
                "details": "There is no code to validate.",
                "fix_suggestions": ["Please provide code to validate."]
            }
        
        try:
            # Create a temporary file to execute the code
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
                temp_file_path = temp_file.name
                # Add necessary imports at the top if they might be missing
                temp_file.write(b"import os\n")
                temp_file.write(b"import sys\n")
                temp_file.write(b"from typing import Dict, Any, List, TypedDict, Optional\n")
                temp_file.write(b"try:\n")
                temp_file.write(b"    from langgraph.graph import StateGraph, START, END\n")
                temp_file.write(b"except ImportError:\n")
                temp_file.write(b"    print('Error: langgraph package is not installed.')\n")
                temp_file.write(b"    sys.exit(1)\n\n")
                
                # Write the generated code
                temp_file.write(code.encode('utf-8'))
                
                # Add a test execution at the bottom
                temp_file.write(b"\n\n# Test execution\n")
                temp_file.write(b"if __name__ == '__main__':\n")
                temp_file.write(b"    try:\n")
                temp_file.write(b"        result = solve('Test problem', 'Test context')\n")
                temp_file.write(b"        print('Code executed successfully!')\n")
                temp_file.write(b"        print(f'Result: {result}')\n")
                temp_file.write(b"        sys.exit(0)\n")
                temp_file.write(b"    except Exception as e:\n")
                temp_file.write(b"        print(f'Error during execution: {str(e)}')\n")
                temp_file.write(b"        traceback.print_exc()\n")
                temp_file.write(b"        sys.exit(1)\n")
            
            # Execute the code as a separate process to safely capture output and errors
            import subprocess
            process = subprocess.Popen(
                [sys.executable, temp_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(timeout=30)  # 30-second timeout
            exit_code = process.returncode
            
            # Clean up the temporary file
            os.unlink(temp_file_path)
            
            if exit_code == 0:
                # Code executed successfully
                return {
                    "valid": True,
                    "output": stdout,
                    "details": "Code executed successfully."
                }
            else:
                # Code execution failed
                error_details = stderr if stderr else stdout
                
                # Analyze the error to provide fix suggestions
                fix_suggestions = self._analyze_error(error_details)
                
                return {
                    "valid": False,
                    "error": "Code execution failed",
                    "details": error_details,
                    "fix_suggestions": fix_suggestions
                }
                
        except subprocess.TimeoutExpired:
            # Handle timeout case
            return {
                "valid": False,
                "error": "Execution timeout",
                "details": "The code execution took too long and was terminated.",
                "fix_suggestions": [
                    "Check for infinite loops in your code.",
                    "Ensure that all async operations are properly awaited.",
                    "Consider optimizing computationally intensive operations."
                ]
            }
        except Exception as e:
            # Handle any other unexpected errors
            tb = traceback.format_exc()
            return {
                "valid": False,
                "error": str(e),
                "details": tb,
                "fix_suggestions": ["The validator itself encountered an error. This might be due to syntax errors in the code."]
            }
    
    def _analyze_error(self, error_details: str) -> List[str]:
        """
        Analyze error details to provide fix suggestions.
        
        Args:
            error_details: The error details as a string.
            
        Returns:
            A list of fix suggestions.
        """
        fix_suggestions = []
        
        # Common error patterns and their fix suggestions
        error_patterns = {
            "ImportError": [
                "Check if all required packages are installed.",
                "Verify import paths and module names.",
                "Ensure required packages are available in the environment."
            ],
            "NameError": [
                "A variable or function name is not defined. Check for typos.",
                "Make sure all variables are defined before use.",
                "Verify that function names are spelled correctly."
            ],
            "TypeError": [
                "Check the types of arguments passed to functions.",
                "Ensure type hints match the actual types used.",
                "Verify operator usage with appropriate types."
            ],
            "AttributeError": [
                "Verify that the object has the attribute or method being accessed.",
                "Check for typos in attribute or method names.",
                "Ensure objects are initialized before accessing their attributes."
            ],
            "KeyError": [
                "Ensure the key exists in the dictionary before accessing it.",
                "Use dict.get(key) method to provide a default value for missing keys.",
                "Check for typos in dictionary keys."
            ],
            "SyntaxError": [
                "Fix the syntax error in the code.",
                "Check for missing parentheses, brackets, or quotes.",
                "Verify indentation is consistent."
            ],
            "IndentationError": [
                "Fix indentation issues in the code.",
                "Use consistent indentation (spaces or tabs, but not both).",
                "Ensure each code block is indented properly."
            ],
            "ValueError": [
                "Check the values passed to functions.",
                "Ensure values are within expected ranges or formats.",
                "Verify string formatting and parsing."
            ],
            "ModuleNotFoundError": [
                "Install the missing module using pip.",
                "Check if the module name is spelled correctly.",
                "Verify that the module is available in your Python environment."
            ],
            "StateGraph": [
                "Ensure you are using the correct import for StateGraph: from langgraph.graph import StateGraph",
                "Verify that the StateGraph is initialized with the correct state type.",
                "Make sure every node in the graph is properly added."
            ],
            "START": [
                "Always add at least one edge from START to another node in your graph.",
                "Check that you have imported START from langgraph.graph.",
                "Verify that START is used as the source in at least one edge."
            ],
            "RuntimeError": [
                "Check for logical errors in your code.",
                "Verify that all required resources are available.",
                "Ensure environmental variables are set correctly."
            ]
        }
        
        # Check which error patterns are present in the error details
        for error_key, suggestions in error_patterns.items():
            if error_key in error_details:
                fix_suggestions.extend(suggestions)
        
        # If no specific error pattern matched, provide general suggestions
        if not fix_suggestions:
            fix_suggestions = [
                "Review the error message carefully for clues about what went wrong.",
                "Check for logical errors in your code.",
                "Verify all function calls and their arguments.",
                "Ensure that the code structure follows the required pattern."
            ]
        
        return fix_suggestions
    
    def display_result(self, result: Dict[str, Any]) -> None:
        """
        Display the validation result in a visually appealing way.
        
        Args:
            result: The validation result.
        """
        if result.get("valid", False):
            console.print(Panel(
                f"[bold green]Code is valid and executes successfully![/]\n\n"
                f"[dim]{result.get('output', '')}[/]",
                title="[bold green]Validation Successful[/]",
                border_style="green",
                expand=False
            ))
        else:
            error = result.get("error", "Unknown error")
            details = result.get("details", "")
            fix_suggestions = result.get("fix_suggestions", [])
            
            suggestions_text = "\n".join([f"• {suggestion}" for suggestion in fix_suggestions])
            
            console.print(Panel(
                f"[bold red]Validation Failed:[/] {error}\n\n"
                f"[yellow]Details:[/]\n{details}\n\n"
                f"[bold green]Fix Suggestions:[/]\n{suggestions_text}",
                title="[bold red]Validation Failed[/]",
                border_style="red",
                expand=False
            )) 