# Deep Search Terminal Agent

A multi-agent system that dynamically solves problems by searching the web and using available knowledge. It orchestrates specialized agents to plan, decompose, build, and execute solutions to complex problems.

## Overview

The Deep Search Terminal Agent is a sophisticated AI orchestration system that:

1. Takes a user problem or question
2. Analyzes and creates a roadmap for solving it
3. Decomposes the roadmap into specialized tasks
4. Generates a dynamic LangGraph execution plan
5. Executes the plan while managing dependencies between steps
6. Displays the results in a visually appealing terminal interface

## Architecture

The system is built on a multi-agent architecture with specialized agents working together:

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│               │     │               │     │               │     │               │     │               │
│    Planner    │ --> │  Decomposer   │ --> │ Graph Builder │ --> │   Validator   │ --> │   Executor    │
│     Agent     │     │     Agent     │     │     Agent     │     │     Agent     │     │     Agent     │
│               │     │               │     │               │     │               │     │               │
└───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘
          │                   │                    │                     │                     │
          └───────────────────┴────────────────────┴─────────────────────┴─────────────────────┘
                                      │
                              ┌───────────────┐
                              │               │
                              │  Orchestrator │
                              │               │
                              └───────────────┘
```

### Agent Roles

- **Planner Agent**: Analyzes the problem and creates a roadmap with steps needed to solve it
- **Decomposer Agent**: Breaks down the roadmap into specialized tasks with dependencies
- **Graph Builder Agent**: Translates tasks into LangGraph code that can be executed
- **Code Validator Agent**: Validates the generated code and provides detailed error feedback
- **Code Regenerator Agent**: Fixes code based on validation errors and suggestions
- **Executor Agent**: Runs the generated code and displays results
- **Orchestrator**: Manages the workflow between agents and maintains conversational context

## Workflow Process

1. **Problem Analysis**:
   - The orchestrator extracts the current problem and any context from previous conversations
   - The planner agent creates a detailed roadmap with steps, alternative approaches, and rationale

2. **Task Decomposition**:
   - The decomposer breaks the roadmap into specialized tasks
   - Each task is assigned an agent type, dependencies, and estimated complexity
   - A workflow graph is created to represent task dependencies

3. **Code Generation**:
   - The graph builder creates LangGraph code that implements the task workflow
   - The code includes proper error handling and resource management

4. **Code Validation**:
   - The validator executes the code in a safe environment and checks for errors
   - If errors are found, the code regenerator fixes the issues based on specific feedback
   - This validation/regeneration cycle repeats until the code is valid or max attempts reached

5. **Execution**:
   - The executor runs the generated code in a controlled environment
   - Results are processed and formatted for display
   - Various output types (tables, markdown, JSON) are handled appropriately

6. **Result Visualization**:
   - Results are displayed in rich, formatted panels in the terminal
   - Complex data structures are presented as tables, syntax-highlighted code, or formatted reports
   - Errors are shown with detailed information for debugging

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Deep-Search-Terminal-Agent.git
cd Deep-Search-Terminal-Agent

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Troubleshooting

### Missing Module Errors

If you encounter errors like `ModuleNotFoundError: No module named 'networkx'` or other missing dependencies, ensure all requirements are installed:

```bash
pip install -r requirements.txt
```

For specific packages, you can install them directly:

```bash
# Install networkx (required for the orchestrator)
pip install networkx

# Install other dependencies as needed
pip install langchain langgraph openai python-dotenv rich
```

### Rich Library Box Error

If you encounter errors related to the Rich library like:
- `Box.__init__() takes 2 positional arguments but 16 were given`
- `module 'rich.box' has no attribute 'box_factory'`

These are due to compatibility issues with different versions of the Rich library. The code has been updated to use standard box styles instead of custom ones. Update your Rich installation:

```bash
pip install rich==13.6.0
```

### Empty Output (None) Error

If the system runs without errors but returns "None" as the result or you see an empty solution panel:

1. The issue is in how the generated code is returning its final result. This has been fixed in the executor agent.

2. If you still encounter the issue, you can manually modify the LangGraph code to return a proper output by editing the `solve` function:

```python
# Run the graph
result = app.invoke({"input": problem, "context": context})

# Return final output instead of the full state
if "key_field" in result:  # Replace key_field with the actual output field
    return result["key_field"]
else:
    return {"output": "No results found", "all_data": result}
```

Make sure your virtual environment is activated when installing packages and running the application.

## Usage

```bash
# Basic usage
python main.py "your problem statement here"

# Start with an initial prompt
python main.py --initial-prompt "Analyze stock market trends for the past week"

# Run in debug mode to see detailed workflow
python main.py --debug
```

### Debug Mode

When running in debug mode (`--debug`), the system will display:
- The detailed problem roadmap
- The task breakdown with complexity estimates
- The generated LangGraph code
- Full execution details and results

## Project Structure

- `main.py`: Entry point with conversation management
- `agents/`: Contains all agent definitions
  - `planner.py`: Creates the initial problem-solving roadmap
  - `decomposer.py`: Breaks down the roadmap into specialized tasks
  - `graph_builder.py`: Generates LangGraph code for execution
  - `executor.py`: Executes generated code and displays results
  - `orchestrator.py`: Manages the full workflow between agents
- `tools/`: Contains tools used by agents for web search and more
- `graph/`: Contains graph definitions and helpers for LangGraph

## Output Formats

The system handles various output formats:

1. **Markdown**: For text-based explanations and reports
2. **Tables**: For structured data with rows and columns
3. **Panels**: For visually distinguishing different sections
4. **Syntax Highlighting**: For code and JSON data
5. **Rich Formatting**: For reports with sections, lists, and highlights

## Environment Variables

Create a `.env` file with:

```
# Azure OpenAI Credentials
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_VERSION=2023-05-15

# Optional Configuration
DEBUG_MODE=False  # Set to True for detailed output
MAX_ATTEMPTS=3    # Number of code validation/regeneration attempts
```

### Configuration Options

- **DEBUG_MODE**: When set to `True`, displays detailed workflow information including roadmaps, task breakdowns, and generated code.
- **MAX_ATTEMPTS**: Controls the maximum number of validation and code regeneration cycles. Default is 3 if not specified.

## Advanced Features

- **Conversation Context**: Maintains context between interactions
- **Error Handling**: Provides detailed error information when execution fails
- **Web Search**: Can search the web for real-time information
- **Task Dependencies**: Manages complex dependencies between tasks
- **Visualization**: Creates visual representations of task workflows
- **Report Formatting**: Intelligently formats complex nested data structures
- **Code Validation**: Validates generated code before execution and provides specific error feedback
- **Auto-Fixing**: Automatically regenerates and fixes code based on validation errors

## How to Extend

To add new capabilities:
1. Create new agent types in `agents/` directory
2. Update the orchestrator to include them in the workflow
3. Add new tools in the `tools/` directory
4. Modify the graph builder to generate code that uses these tools

## License

[Your License Here] 