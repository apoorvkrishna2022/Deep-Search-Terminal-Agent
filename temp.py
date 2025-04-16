import os
from dotenv import load_dotenv
# Load environment variables first
load_dotenv()

from typing import Dict, Any, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import AzureChatOpenAI
from tools.web_search import WebSearchTool

# Define state schema
class AgentState(TypedDict):
    query: str
    context: Optional[str]
    defined_scope: Optional[str]
    search_engine_info: Optional[str]
    google_assistant_info: Optional[str]
    synthesized_explanation: Optional[str]
    final_presentation: Optional[str]

# Create agents
def define_scope(state: AgentState) -> Dict[str, Any]:
    # For simplicity, assume the user is interested in both search engine and Google Assistant
    return {"defined_scope": "search_engine_and_assistant"}

def conduct_web_search_on_search_engine(state: AgentState) -> Dict[str, Any]:
    if state.get('defined_scope') == "search_engine_and_assistant":
        search_tool = WebSearchTool()
        search_results = search_tool.run("how does Google's search engine work")
        return {"search_engine_info": search_results}
    return {}

def conduct_web_search_on_google_assistant(state: AgentState) -> Dict[str, Any]:
    if state.get('defined_scope') == "search_engine_and_assistant":
        search_tool = WebSearchTool()
        search_results = search_tool.run("how does Google Assistant work")
        return {"google_assistant_info": search_results}
    return {}

def synthesize_information(state: AgentState) -> Dict[str, Any]:
    search_info = state.get('search_engine_info', '')
    assistant_info = state.get('google_assistant_info', '')
    synthesized = f"Synthesized Information:\n\nSearch Engine:\n{search_info}\n\nGoogle Assistant:\n{assistant_info}"
    return {"synthesized_explanation": synthesized}

def present_findings(state: AgentState) -> Dict[str, Any]:
    return {"final_presentation": state.get('synthesized_explanation', 'No information available.')}

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
    workflow.add_node("define_scope", define_scope)
    workflow.add_node("conduct_web_search_on_search_engine", conduct_web_search_on_search_engine)
    workflow.add_node("conduct_web_search_on_google_assistant", conduct_web_search_on_google_assistant)
    workflow.add_node("synthesize_information", synthesize_information)
    workflow.add_node("present_findings", present_findings)

    # Add edges
    workflow.add_edge(START, "define_scope")
    workflow.add_edge("define_scope", "conduct_web_search_on_search_engine")
    workflow.add_edge("define_scope", "conduct_web_search_on_google_assistant")
    
    # Use 'and' condition for multiple inputs to synthesize_information
    workflow.add_edge(
        ["conduct_web_search_on_search_engine", "conduct_web_search_on_google_assistant"],
        "synthesize_information"
    )
    
    workflow.add_edge("synthesize_information", "present_findings")
    workflow.add_edge("present_findings", END)

    # Compile
    app = workflow.compile()

    # Run the graph
    result = app.invoke({"query": problem, "context": context})

    # Return final output instead of the full state
    if "final_presentation" in result:
        return {"final_presentation": result["final_presentation"]}
    else:
        return {"error": "Failed to generate a presentation."}

# Example usage
if __name__ == "__main__":
    problem_statement = "hey how does google work?"
    previous_context = "Previous question: hey how does google work?"
    output = solve(problem_statement, previous_context)
    print(output)