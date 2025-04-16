"""
Executor agent responsible for running the generated LangGraph code.
"""

from typing import Dict, Any, List
import tempfile
import os
import sys
import importlib.util
import traceback
import inspect
import json
import re
import uuid
import logging
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import box
from rich.syntax import Syntax

from agents.base import BaseAgent, AgentConfig
from graph.fallback_searcher import solve as fallback_solve

SYSTEM_PROMPT = """You are a code execution specialist. Your job is to execute Python code and provide the results.

When given Python code, you should:
1. Execute the code safely
2. Capture all outputs and errors
3. Format the results in a clear, readable way
4. Provide helpful context about any errors or issues
5. Pass any relevant conversation context to the executed code

You are specifically designed to run LangGraph workflows and report on their execution.
"""

console = Console()

class ExecutorAgent(BaseAgent):
    """
    Agent responsible for executing the generated code to solve the problem.
    """
    
    def __init__(self, **kwargs):
        """Initialize the executor agent."""
        config = AgentConfig(
            name="Code Executor",
            description="Safely executes generated code and reports results",
            system_prompt=SYSTEM_PROMPT,
            **kwargs
        )
        super().__init__(config=config)
    
    def process_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the execution result and format it for better readability.
        
        Args:
            result: The raw execution result.
            
        Returns:
            Formatted and processed result.
        """
        if not result or not isinstance(result, dict):
            return {"success": False, "error": "Invalid result format", "details": "Expected a dictionary result."}
        
        # For successful results, format the output
        if result.get("success", False) and "output" in result:
            output = result["output"]
            
            # If the output is a dictionary, format it nicely
            if isinstance(output, dict):
                formatted_output = self._format_dict_output(output)
                result["output"] = formatted_output
                # Also store the display version for terminal output
                result["display_output"] = formatted_output
            
        return result
    
    def _format_dict_output(self, output: Dict[str, Any]) -> Table:
        """
        Format a dictionary output as a table.
        
        Args:
            output: The dictionary to format.
            
        Returns:
            A formatted Rich Table.
        """
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in output.items():
            formatted_key = " ".join(word.capitalize() for word in key.split('_'))
            
            # Handle nested dictionaries or lists by pretty formatting them
            if isinstance(value, dict):
                # For nested dictionaries, create a nested table or format as JSON
                formatted_value = json.dumps(value, indent=2)
                table.add_row(formatted_key, Panel(Syntax(formatted_value, "json", theme="monokai", line_numbers=False)))
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                # For lists of dictionaries (data points), create a nice representation
                formatted_value = json.dumps(value, indent=2)
                table.add_row(formatted_key, Panel(Syntax(formatted_value, "json", theme="monokai", line_numbers=False)))
            elif isinstance(value, list):
                # Format simple lists with bullet points
                formatted_value = "\n".join([f"• {item}" for item in value])
                table.add_row(formatted_key, formatted_value)
            else:
                # Simple string representation for other types
                table.add_row(formatted_key, str(value))
        
        return table
    
    def _format_report_output(self, output: Dict[str, Any]) -> Panel:
        """
        Format a report-style output with sections.
        
        Args:
            output: The report data to format.
            
        Returns:
            A formatted Rich Panel.
        """
        # Detect if this looks like a report with sections
        if not isinstance(output, dict) or not any(k in output for k in ['summary', 'findings', 'conclusion', 'recommendations']):
            return None
            
        # Create a report layout
        report = ""
        
        # Add title if available
        if 'title' in output:
            report += f"[bold cyan]{output['title']}[/]\n\n"
            
        # Add summary section
        if 'summary' in output:
            report += f"[bold]Summary:[/]\n{output['summary']}\n\n"
            
        # Add findings or main content
        if 'findings' in output:
            report += f"[bold]Key Findings:[/]\n"
            if isinstance(output['findings'], list):
                for i, finding in enumerate(output['findings'], 1):
                    if isinstance(finding, dict) and 'point' in finding:
                        report += f"{i}. [yellow]{finding.get('title', f'Finding {i}')}[/]\n"
                        report += f"   {finding['point']}\n\n"
                    else:
                        report += f"{i}. {finding}\n"
            else:
                report += f"{output['findings']}\n\n"
                
        # Add any other sections that exist
        for section in ['details', 'methodology', 'data', 'analysis']:
            if section in output:
                section_title = section.capitalize()
                report += f"[bold]{section_title}:[/]\n{output[section]}\n\n"
        
        # Add conclusion
        if 'conclusion' in output:
            report += f"[bold]Conclusion:[/]\n{output['conclusion']}\n\n"
            
        # Add recommendations if available
        if 'recommendations' in output:
            report += f"[bold]Recommendations:[/]\n"
            if isinstance(output['recommendations'], list):
                for i, rec in enumerate(output['recommendations'], 1):
                    report += f"{i}. {rec}\n"
            else:
                report += f"{output['recommendations']}\n"
        
        return Panel(
            Markdown(report),
            title="[bold green]Research Report[/]",
            border_style="green",
            box=box.ROUNDED,
            expand=False
        )
    
    def display_result(self, result: Dict[str, Any]) -> None:
        """
        Display the execution result in a visually appealing way.
        
        Args:
            result: The execution result.
        """
        if result.get("success", False):
            output = result.get("output", "No output was produced.")
            
            # Handle None output explicitly
            if output is None:
                # Check if there are other fields with useful information
                other_data = {}
                if isinstance(result, dict):
                    for key, value in result.items():
                        if key not in ["success", "output"] and value is not None:
                            other_data[key] = value
                
                if other_data:
                    # Create a table for the other data
                    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
                    table.add_column("Key", style="cyan")
                    table.add_column("Value", style="green")
                    for key, value in other_data.items():
                        formatted_key = " ".join(word.capitalize() for word in key.split('_'))
                        table.add_row(formatted_key, str(value))
                    
                    console.print(Panel(
                        table,
                        title="[bold green]Solution[/]",
                        border_style="green",
                        box=box.ROUNDED,
                        expand=False
                    ))
                else:
                    console.print(Panel(
                        "The execution completed successfully, but no output was produced.",
                        title="[bold green]Solution[/]",
                        border_style="green",
                        box=box.ROUNDED,
                        expand=False
                    ))
                return
            
            # If the output is a string that looks like markdown, render it as markdown
            if isinstance(output, str) and ('##' in output or '#' in output):
                console.print(Panel(
                    Markdown(output),
                    title="[bold green]Solution[/]",
                    border_style="green",
                    box=box.ROUNDED,
                    expand=False
                ))
            elif isinstance(output, dict):
                # Try to format as a report if it looks like one
                report_panel = self._format_report_output(output)
                if report_panel:
                    console.print(report_panel)
                else:
                    # Create a nice table for dictionary output
                    table = self._format_dict_output(output)
                    console.print(Panel(
                        table,
                        title="[bold green]Solution[/]",
                        border_style="green",
                        box=box.ROUNDED,
                        expand=False
                    ))
            elif isinstance(output, list) and output and isinstance(output[0], dict):
                # For lists of dictionaries, create a data table
                if all('name' in item and 'value' in item for item in output):
                    # This looks like a name-value pair list
                    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
                    table.add_column("Name", style="cyan")
                    table.add_column("Value", style="green")
                    for item in output:
                        table.add_row(item['name'], str(item['value']))
                    console.print(Panel(
                        table,
                        title="[bold green]Solution[/]",
                        border_style="green",
                        box=box.ROUNDED,
                        expand=False
                    ))
                else:
                    # Generic list of dictionaries
                    console.print(Panel(
                        Syntax(json.dumps(output, indent=2), "json", theme="monokai"),
                        title="[bold green]Solution[/]",
                        border_style="green",
                        box=box.ROUNDED,
                        expand=False
                    ))
            else:
                # Simple string output
                console.print(Panel(
                    str(output),
                    title="[bold green]Solution[/]",
                    border_style="green",
                    box=box.ROUNDED,
                    expand=False
                ))
        else:
            error_msg = result.get("error", "Unknown error")
            details = result.get("details", "")
            traceback_info = result.get("traceback", "")
            
            error_panel = Panel(
                f"[bold red]Error:[/] {error_msg}\n\n"
                f"[yellow]Details:[/] {details}\n\n"
                + (f"[dim]Traceback:[/]\n{traceback_info}" if traceback_info else ""),
                title="[bold red]Execution Failed[/]",
                border_style="red",
                box=box.ROUNDED,
                expand=False
            )
            console.print(error_panel)
    
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the executor agent.
        
        Args:
            inputs: The inputs for the agent.
            
        Returns:
            The execution result.
        """
        # Get the generated code
        code = inputs.get("code", "")
        problem = inputs.get("problem", "")
        context = inputs.get("context", "")
        
        if not code:
            return {
                "result": {
                    "success": False,
                    "error": "No code provided",
                    "details": "The graph builder did not provide any code to execute."
                }
            }
        
        # Create a function name for the generated code
        func_name = f"solve_{uuid.uuid4().hex[:8]}"
        
        # Define the full code with input
        full_code = (
            f"def {func_name}(problem, context):\n"
            f"    '''\n"
            f"    Execute the solution to the problem.\n"
            f"    \n"
            f"    Args:\n"
            f"        problem: The problem statement.\n"
            f"        context: The context for the problem.\n"
            f"    \n"
            f"    Returns:\n"
            f"        The result of executing the solution.\n"
            f"    '''\n"
            + "\n".join([f"    {line}" for line in code.split("\n")])
            + "\n"
        )
        
        try:
            # Create a namespace to execute the code
            namespace = {}
            
            # Execute the code to define the function
            exec(full_code, globals(), namespace)
            
            # Get the defined function
            solve_func = namespace[func_name]
            
            # Execute the function
            result = solve_func(problem, context)
            
            # Process and format the result
            processed_result = {}
            
            # Special handling for LangGraph state objects which are usually dictionaries
            if isinstance(result, dict):
                # Flag to track if we've found meaningful content
                has_meaningful_content = False
                meaningful_data = {}
                
                # First check for common LangGraph result patterns
                if "__final_outputs" in result:
                    # This is a typical LangGraph format, extract the final values
                    meaningful_data["output"] = result["__final_outputs"]
                    has_meaningful_content = True
                else:
                    # Extract meaningful fields from the state
                    for key, value in result.items():
                        # Skip input/context fields and None values
                        if key not in ["input", "context"] and value is not None:
                            meaningful_data[key] = value
                            has_meaningful_content = True
                
                if has_meaningful_content:
                    processed_result = self.process_result({
                        "success": True,
                        "output": meaningful_data
                    })
                else:
                    # If no meaningful data found, return the whole result as structured data
                    processed_result = self.process_result({
                        "success": True,
                        "output": result
                    })
            else:
                # For non-dictionary results, process as is
                processed_result = self.process_result({
                    "success": True,
                    "output": result
                })
            
            # Display the result in a visually appealing way
            self.display_result(processed_result)
            
            return {"result": processed_result}
            
        except Exception as e:
            # Get the traceback
            tb = traceback.format_exc()
            
            error_result = {
                "success": False,
                "error": str(e),
                "details": "An error occurred while executing the generated code.",
                "traceback": tb
            }
            
            # Display the error in a visually appealing way
            self.display_result(error_result)
            
            return {"result": error_result} 