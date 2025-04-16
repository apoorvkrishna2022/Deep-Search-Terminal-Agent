#!/usr/bin/env python3
"""
Deep Search Terminal Agent
A multi-agent system that solves problems through web search and knowledge.
"""

import os
import sys
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown

# Load local modules
from agents.orchestrator import Orchestrator

# Initialize console for rich output
console = Console()

def setup_environment():
    """Setup environment variables for LangChain and Azure OpenAI."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Set default environment variables if not already set
    if not os.getenv("LANGCHAIN_TRACING_V2"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "deep-search-terminal-agent"
    
    # Check if Azure OpenAI API key is set
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        console.print("[bold red]Warning:[/] AZURE_OPENAI_API_KEY is not set in the .env file")
        console.print("Please set up your Azure OpenAI credentials in the .env file")
        return False
    
    # Verify other necessary Azure OpenAI environment variables
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        console.print(f"[bold red]Warning:[/] The following required environment variables are not set: {', '.join(missing_vars)}")
        console.print("Please set up your Azure OpenAI credentials in the .env file")
        return False
    
    # Set MAX_ATTEMPTS if not provided or convert to int
    if os.getenv("MAX_ATTEMPTS") is None:
        os.environ["MAX_ATTEMPTS"] = "3"
        console.print("[yellow]Notice:[/] MAX_ATTEMPTS not defined in .env, using default value of 3")
    else:
        try:
            # Ensure MAX_ATTEMPTS is a valid integer
            max_attempts = int(os.getenv("MAX_ATTEMPTS"))
            if max_attempts < 1:
                console.print("[bold yellow]Warning:[/] MAX_ATTEMPTS must be at least 1, setting to default of 3")
                os.environ["MAX_ATTEMPTS"] = "3"
        except ValueError:
            console.print("[bold yellow]Warning:[/] MAX_ATTEMPTS must be an integer, setting to default of 3")
            os.environ["MAX_ATTEMPTS"] = "3"
    
    return True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deep Search Terminal Agent - Solve problems through multi-agent collaboration."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--initial-prompt",
        type=str,
        help="Initial problem statement to start with",
    )
    return parser.parse_args()

class ConversationalAgent:
    """Manages a continuous conversation with the user, maintaining context between interactions."""
    
    def __init__(self, debug: bool = False):
        """
        Initialize the conversational agent.
        
        Args:
            debug: Whether to enable debug mode.
        """
        self.debug = debug
        
        # Get MAX_ATTEMPTS from environment
        max_attempts = int(os.getenv("MAX_ATTEMPTS", "3"))
        
        # Initialize the orchestrator with MAX_ATTEMPTS from environment
        self.orchestrator = Orchestrator(debug=debug, max_code_attempts=max_attempts)
        self.conversation_history = []
        self.show_welcome_message()
    
    def show_welcome_message(self):
        """Display a welcome message."""
        console.print(Panel.fit(
            "Welcome to the Deep Search Terminal Agent\n"
            "Type your problem or question, and I'll solve it for you.\n"
            "Type 'exit', 'quit', or 'bye' to end the conversation.",
            title="Deep Search Terminal Agent",
            border_style="blue",
        ))
    
    def add_to_history(self, role: str, content: str):
        """
        Add a message to the conversation history.
        
        Args:
            role: The role of the message sender (user/assistant).
            content: The content of the message.
        """
        self.conversation_history.append({"role": role, "content": content})
    
    def get_context_summary(self) -> str:
        """
        Get a summary of the conversation context.
        
        Returns:
            A string summarizing the conversation context.
        """
        if not self.conversation_history:
            return ""
        
        # Create a summary of previous interactions
        summary_parts = []
        for i, message in enumerate(self.conversation_history[-4:]):  # Last 4 messages
            if message["role"] == "user":
                summary_parts.append(f"Previous question: {message['content']}")
            else:
                # Truncate long responses
                content = message["content"]
                if len(content) > 200:
                    content = content[:197] + "..."
                summary_parts.append(f"Previous answer: {content}")
        
        if summary_parts:
            return "Previous conversation context:\n" + "\n".join(summary_parts)
        return ""
    
    def start_conversation(self, initial_prompt: str = None):
        """
        Start a conversation loop with the user.
        
        Args:
            initial_prompt: Optional initial problem to solve.
        """
        # If we have an initial prompt, process it first
        if initial_prompt:
            self.process_input(initial_prompt)
        
        # Start the conversation loop
        while True:
            try:
                # Get input from the user
                user_input = Prompt.ask("\n[bold green]What problem would you like me to solve?[/]")
                
                # Check if the user wants to exit
                if user_input.lower() in ["exit", "quit", "bye"]:
                    console.print("[bold blue]Thank you for using Deep Search Terminal Agent. Goodbye![/]")
                    break
                
                # Process the user input
                self.process_input(user_input)
                
            except KeyboardInterrupt:
                console.print("\n[bold red]Conversation interrupted by user.[/]")
                break
            except Exception as e:
                console.print(f"\n[bold red]Error:[/] {str(e)}")
                if self.debug:
                    raise
    
    def process_input(self, user_input: str):
        """
        Process a user input and generate a response.
        
        Args:
            user_input: The user's input/problem statement.
        """
        # Add the user input to the conversation history
        self.add_to_history("user", user_input)
        
        # Get the conversation context
        context = self.get_context_summary()
        
        # Create the full problem with context
        full_problem = user_input
        if context:
            full_problem = f"{context}\n\nCurrent problem: {user_input}"
        
        # Display the problem being worked on
        console.print(f"[bold]Working on problem:[/] {user_input}")
        
        # Solve the problem
        solution = self.orchestrator.solve(full_problem)
        
        # Add the solution to the conversation history
        self.add_to_history("assistant", solution)
        
        # Display the solution
        console.print(Panel.fit(
            Markdown(solution),
            title="Solution",
            border_style="green",
        ))

def main():
    """Main entry point for the application."""
    # Setup environment variables
    if not setup_environment():
        sys.exit(1)
    
    # Parse arguments
    args = parse_args()
    
    # Set debug mode
    debug_mode = args.debug or os.getenv("DEBUG_MODE", "False").lower() == "true"
    
    # Initialize the conversational agent
    conversation_agent = ConversationalAgent(debug=debug_mode)
    
    # Start the conversation
    conversation_agent.start_conversation(initial_prompt=args.initial_prompt)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("[bold red]Process interrupted by user.[/]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        if os.getenv("DEBUG_MODE", "False").lower() == "true":
            raise
        sys.exit(1) 