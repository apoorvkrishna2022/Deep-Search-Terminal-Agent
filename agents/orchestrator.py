"""
Orchestrator agent responsible for managing the entire workflow.
"""

from typing import Dict, Any, List
import os
import re
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from rich.table import Table
from rich.markdown import Markdown
from rich.text import Text
from rich.box import Box, ROUNDED
from rich import box
import networkx as nx
from collections import defaultdict

from agents.base import BaseAgent
from agents.planner import PlannerAgent
from agents.decomposer import DecomposerAgent
from agents.graph_builder import GraphBuilderAgent
from agents.executor import ExecutorAgent
from agents.code_validator import CodeValidatorAgent
from agents.code_regenerator import CodeRegeneratorAgent

console = Console()

class Orchestrator:
    """
    Orchestrator that manages the entire workflow from problem statement to solution.
    """
    
    def __init__(self, debug: bool = False, max_code_attempts: int = None):
        """
        Initialize the orchestrator.
        
        Args:
            debug: Whether to enable debug mode.
            max_code_attempts: Maximum number of attempts to fix code. If None, read from environment.
        """
        self.debug = debug
        
        # Get max code attempts from environment or use default
        if max_code_attempts is None:
            self.max_code_attempts = int(os.getenv("MAX_ATTEMPTS", "3"))
        else:
            self.max_code_attempts = max_code_attempts
        
        # Initialize agents
        self.planner = PlannerAgent()
        self.decomposer = DecomposerAgent()
        self.graph_builder = GraphBuilderAgent()
        self.validator = CodeValidatorAgent()
        self.regenerator = CodeRegeneratorAgent()
        self.executor = ExecutorAgent()
    
    def extract_current_problem(self, full_problem: str) -> tuple:
        """
        Extract the current problem from a potentially context-enriched problem statement.
        
        Args:
            full_problem: The full problem statement, possibly with conversation context.
            
        Returns:
            A tuple of (current_problem, context)
        """
        # Check if the problem has our context format
        if "Current problem:" in full_problem:
            # Split the context and current problem
            parts = full_problem.split("Current problem:", 1)
            context = parts[0].strip()
            current_problem = parts[1].strip()
            return current_problem, context
        else:
            # No specific format, treat the whole input as the current problem
            return full_problem, ""
    
    def _display_roadmap(self, roadmap: Dict[str, Any]) -> None:
        """
        Display the roadmap in a visually appealing way using Rich formatting.
        
        Args:
            roadmap: The roadmap dictionary to display.
        """
        if not roadmap:
            return
        
        console.print("\n")
        
        # Problem analysis panel
        if "problem_analysis" in roadmap:
            problem_analysis = roadmap["problem_analysis"]
            console.print(Panel(
                Text(problem_analysis, style="blue"),
                title="[bold cyan]Problem Analysis[/]",
                border_style="cyan",
                expand=False
            ))
        
        # Roadmap steps as a table
        if "roadmap" in roadmap and roadmap["roadmap"]:
            console.print("\n[bold cyan]Roadmap Steps[/]")
            
            table = Table(show_header=True, header_style="bold magenta", expand=True, box=box.ROUNDED)
            table.add_column("Step", style="dim", width=4)
            table.add_column("Title", style="cyan")
            table.add_column("Description", style="green", max_width=40, overflow="fold")
            table.add_column("Tools", style="yellow")
            
            for step in roadmap["roadmap"]:
                tools = ", ".join(step.get("tools_needed", []))
                table.add_row(
                    str(step.get("step", "")),
                    step.get("title", ""),
                    step.get("description", ""),
                    tools or "None"
                )
            
            console.print(table)
        
        # Alternative approaches
        if "alternative_approaches" in roadmap and roadmap["alternative_approaches"]:
            console.print("\n[bold cyan]Alternative Approaches[/]")
            
            for i, approach in enumerate(roadmap["alternative_approaches"]):
                approach_panel = Panel(
                    f"[green]Pros:[/]\n" + "\n".join([f"• {pro}" for pro in approach.get("pros", [])]) +
                    f"\n\n[red]Cons:[/]\n" + "\n".join([f"• {con}" for con in approach.get("cons", [])]),
                    title=f"[bold]{approach.get('approach', f'Approach {i+1}')}[/]",
                    border_style="blue",
                    expand=False,
                    box=box.ROUNDED
                )
                console.print(approach_panel)
        
        # Rationale
        if "rationale" in roadmap:
            console.print(Panel(
                Text(roadmap["rationale"], style="yellow"),
                title="[bold cyan]Rationale[/]",
                border_style="cyan",
                expand=False,
                box=box.ROUNDED
            ))
        
        console.print("\n")
    
    def _display_task_breakdown(self, task_breakdown: Dict[str, Any]) -> None:
        """
        Display the task breakdown in a visually appealing way using Rich formatting.
        
        Args:
            task_breakdown: The task breakdown dictionary to display.
        """
        if not task_breakdown:
            return
        
        console.print("\n")
        
        # Agent types
        if "agent_types" in task_breakdown and task_breakdown["agent_types"]:
            console.print("[bold cyan]Agent Types[/]")
            
            agent_table = Table(show_header=True, header_style="bold magenta", expand=True, box=box.ROUNDED)
            agent_table.add_column("Type", style="cyan")
            agent_table.add_column("Description", style="green")
            agent_table.add_column("Capabilities", style="yellow")
            agent_table.add_column("Tools", style="blue")
            
            for agent in task_breakdown["agent_types"]:
                capabilities = ", ".join(agent.get("capabilities", []))
                tools = ", ".join(agent.get("tools", []))
                
                agent_table.add_row(
                    agent.get("type", ""),
                    agent.get("description", ""),
                    capabilities,
                    tools
                )
            
            console.print(agent_table)
        
        # Create a visual graph representation of tasks
        if "tasks" in task_breakdown and task_breakdown["tasks"]:
            console.print("\n[bold cyan]Task Workflow Graph[/]")
            
            # Build the graph using ASCII-based visualization
            self._display_task_graph(task_breakdown)
            
            # Detailed task information
            console.print("\n[bold cyan]Task Details[/]")
            
            for task in task_breakdown.get("tasks", []):
                task_id = task.get("id", "unknown")
                title = task.get("title", "Untitled Task")
                description = task.get("description", "")
                agent_type = task.get("agent_type", "Unknown")
                dependencies = ", ".join(task.get("dependencies", []))
                tools = ", ".join(task.get("tools", []))
                complexity = task.get("estimated_complexity", "unknown")
                
                # Use different border colors based on complexity
                border_style = "green"
                if complexity == "medium":
                    border_style = "yellow"
                elif complexity == "high":
                    border_style = "red"
                
                # Create a panel for each task
                task_panel = Panel(
                    f"[white]Description:[/] {description}\n\n"
                    f"[white]Agent Type:[/] [cyan]{agent_type}[/]\n"
                    f"[white]Dependencies:[/] {dependencies or 'None'}\n"
                    f"[white]Tools:[/] {tools or 'None'}\n"
                    f"[white]Complexity:[/] [{border_style}]{complexity}[/]",
                    title=f"[bold]{title}[/] ([dim]{task_id}[/])",
                    border_style=border_style,
                    expand=False,
                    box=box.ROUNDED
                )
                
                console.print(task_panel)
        
        # Tool requests
        if "tool_requests" in task_breakdown and task_breakdown["tool_requests"]:
            console.print("\n[bold cyan]Tool Requests[/]")
            
            for tool_request in task_breakdown["tool_requests"]:
                status = tool_request.get("status", "requested")
                status_style = "green" if status == "already_available" else "yellow"
                
                tool_panel = Panel(
                    f"[white]Description:[/] {tool_request.get('description', '')}\n\n"
                    f"[white]Reason needed:[/] {tool_request.get('reason_needed', '')}\n\n"
                    f"[white]Workaround:[/] {tool_request.get('workaround', 'No workaround specified')}\n\n"
                    f"[white]Status:[/] [{status_style}]{status}[/]",
                    title=f"[bold]{tool_request.get('name', 'Unnamed Tool')}[/]",
                    border_style="blue",
                    expand=False,
                    box=box.ROUNDED
                )
                console.print(tool_panel)
        
        console.print("\n")
    
    def _display_task_graph(self, task_breakdown: Dict[str, Any]) -> None:
        """
        Display a visual graph representation of the task workflow.
        
        Args:
            task_breakdown: The task breakdown dictionary.
        """
        tasks = task_breakdown.get("tasks", [])
        if not tasks:
            return
        
        # Create the task dictionary for quick lookups
        task_dict = {}
        for task in tasks:
            task_id = task.get("id", "")
            if task_id:
                task_dict[task_id] = task
        
        # Create adjacency list representation of the graph
        graph = defaultdict(list)
        for task in tasks:
            task_id = task.get("id", "")
            for dep_id in task.get("dependencies", []):
                if dep_id in task_dict:
                    graph[dep_id].append(task_id)
        
        # Get initial and terminal tasks
        workflow = task_breakdown.get("workflow", {})
        initial_tasks = workflow.get("initial_tasks", [])
        terminal_tasks = workflow.get("terminal_tasks", [])
        
        # Create a visual ASCII graph
        console.print("\n[bold]Task Graph ([blue]→[/] indicates task flow)[/]\n")
        
        # Draw the graph level by level using BFS
        queue = [(task_id, 0) for task_id in initial_tasks]  # (task_id, level)
        visited = set()
        max_level = 0
        level_tasks = defaultdict(list)
        
        while queue:
            task_id, level = queue.pop(0)
            if task_id in visited:
                continue
            
            visited.add(task_id)
            level_tasks[level].append(task_id)
            max_level = max(max_level, level)
            
            for child_id in graph.get(task_id, []):
                if child_id not in visited:
                    queue.append((child_id, level + 1))
        
        # Custom box for visual graph representation
        # Use a predefined box instead of trying to create a custom one
        custom_box = box.SIMPLE
        
        # Draw the graph
        for level in range(max_level + 1):
            tasks_at_level = level_tasks[level]
            if not tasks_at_level:
                continue
            
            # Create a panel for each level with tasks
            level_panels = []
            
            for task_id in tasks_at_level:
                if task_id in task_dict:
                    task = task_dict[task_id]
                    title = task.get("title", task_id)
                    agent = task.get("agent_type", "Unknown")
                    
                    # Determine node style based on position in workflow
                    node_style = "yellow"
                    if task_id in initial_tasks:
                        node_style = "green"
                    elif task_id in terminal_tasks:
                        node_style = "red"
                    
                    # Create a panel for the task
                    task_panel = Panel(
                        f"[bold]{title}[/]\n[dim]{task_id}[/]\n[cyan]{agent}[/]",
                        box=custom_box,
                        border_style=node_style,
                        width=30,
                        padding=(0, 1)
                    )
                    level_panels.append(task_panel)
            
            # Display the tasks at this level
            console.print("[bold]Level " + str(level) + "[/]")
            
            # Use a table for horizontal alignment
            tasks_table = Table.grid(padding=2)
            tasks_table.add_row(*level_panels)
            console.print(tasks_table)
            
            # Show connections to next level if there are any
            if level < max_level:
                connections = []
                
                for task_id in tasks_at_level:
                    children = graph.get(task_id, [])
                    if children:
                        connections.append(f"[dim]{task_id}[/] [blue]→[/] {', '.join([f'[dim]{c}[/]' for c in children])}")
                
                if connections:
                    connections_panel = Panel(
                        "\n".join(connections),
                        title="[bold blue]Connections[/]",
                        border_style="blue",
                        padding=(0, 1)
                    )
                    console.print(connections_panel)
        
        console.print("\n[bold]Key:[/] [green]Initial Task[/] | [yellow]Intermediate Task[/] | [red]Terminal Task[/]\n")
    
    def _add_dependent_tasks(self, parent_node, parent_id, task_dict, task_breakdown):
        """
        Recursively add dependent tasks to the tree.
        
        Args:
            parent_node: The parent tree node.
            parent_id: The ID of the parent task.
            task_dict: Dictionary of all tasks.
            task_breakdown: The complete task breakdown.
        """
        # Find all tasks that depend on this task
        dependent_tasks = []
        for task in task_breakdown.get("tasks", []):
            if "dependencies" in task and parent_id in task.get("dependencies", []):
                dependent_tasks.append(task)
        
        for task in dependent_tasks:
            task_id = task.get("id", "")
            if not task_id:
                continue
                
            task_node = parent_node.add(f"[bold cyan]{task.get('title', task_id)}[/] ([yellow]{task.get('agent_type', 'Unknown')}[/])")
            task_node.add(f"[dim]ID:[/] {task_id}")
            task_node.add(f"[green]{task.get('description', '')}[/]")
            
            # Recursively add dependent tasks
            self._add_dependent_tasks(task_node, task_id, task_dict, task_breakdown)
    
    def solve(self, problem: str) -> str:
        """
        Solve the given problem using the multi-agent system.
        
        Args:
            problem: The problem statement to solve.
            
        Returns:
            The solution to the problem.
        """
        # Extract the current problem and context
        current_problem, context = self.extract_current_problem(problem)
        
        # Step 1: Create a roadmap using the planner
        console.print("[bold blue]Step 1:[/] Creating problem-solving roadmap...")
        planner_result = self.planner.run({"problem": current_problem, "context": context})
        roadmap = planner_result.get("roadmap", {})
        
        # Display the roadmap in a visually appealing way
        if self.debug:
            self._display_roadmap(roadmap)
        
        # Step 2: Break down the roadmap into tasks using the decomposer
        console.print("[bold blue]Step 2:[/] Breaking down roadmap into specialized tasks...")
        decomposer_result = self.decomposer.run({"roadmap": roadmap, "context": context})
        task_breakdown = decomposer_result.get("task_breakdown", {})
        
        # Display the task breakdown in a visually appealing way
        if self.debug:
            self._display_task_breakdown(task_breakdown)
        
        # Step 3: Create LangGraph code using the graph builder
        console.print("[bold blue]Step 3:[/] Generating LangGraph code...")
        graph_builder_result = self.graph_builder.run({
            "problem": current_problem,
            "context": context,
            "roadmap": roadmap,
            "task_breakdown": task_breakdown
        })
        code = graph_builder_result.get("code", "")
        
        if self.debug:
            console.print(Panel(
                Markdown(f"```python\n{code}\n```"),
                title="[bold blue]Generated Code[/]",
                border_style="blue",
                expand=False
            ))
        
        # Step 3.5: Validate and fix code in a loop until it passes or max attempts reached
        console.print("[bold blue]Step 3.5:[/] Validating and fixing code...")
        code_is_valid = False
        code_attempt = 1
        validation_result = None
        
        while not code_is_valid and code_attempt <= self.max_code_attempts:
            if code_attempt > 1:
                console.print(f"[bold yellow]Code validation attempt {code_attempt}/{self.max_code_attempts}[/]")
            
            # Validate the code
            validation_result = self.validator.run({
                "code": code,
                "problem": current_problem,
                "context": context
            })
            
            # Display validation result
            self.validator.display_result(validation_result)
            
            # Check if code is valid
            if validation_result.get("valid", False):
                code_is_valid = True
                console.print("[bold green]Code successfully validated![/]")
                break
            
            # If we've hit the max attempts, break out of the loop
            if code_attempt >= self.max_code_attempts:
                console.print(f"[bold red]Failed to fix code after {self.max_code_attempts} attempts.[/]")
                break
                
            # Regenerate the code with fixes
            console.print("[bold blue]Regenerating code with fixes...[/]")
            regenerator_result = self.regenerator.run({
                "code": code,
                "validation_result": validation_result,
                "problem": current_problem,
                "context": context,
                "attempt_number": code_attempt
            })
            
            # Update the code with the fixed version
            code = regenerator_result.get("code", code)
            
            if self.debug:
                console.print(Panel(
                    Markdown(f"```python\n{code}\n```"),
                    title=f"[bold blue]Fixed Code (Attempt {code_attempt})[/]",
                    border_style="blue",
                    expand=False
                ))
                
            # Increment the attempt counter
            code_attempt += 1
        
        # Step 4: Execute the generated code using the executor
        console.print("[bold blue]Step 4:[/] Executing solution workflow...")
        executor_result = self.executor.run({
            "code": code,
            "problem": current_problem,
            "context": context
        })
        result = executor_result.get("result", {})
        
        # Let the executor handle displaying the result
        if not self.debug:  # Only if not already displayed during execution
            self.executor.display_result(result)
        
        # If validation failed but we're executing anyway, show a warning
        if not code_is_valid:
            console.print("[bold yellow]Warning:[/] Executing code that did not pass validation.")
            
            if validation_result:
                error = validation_result.get("error", "Unknown error")
                console.print(f"[yellow]Last validation error: {error}[/]")
        
        # Process the result for return value
        if result.get("success", False):
            output = result.get("output", "No output was produced.")
            return str(output)
        else:
            error = result.get("error", "Unknown error")
            details = result.get("details", "")
            return f"Failed to solve the problem: {error}. {details}" 