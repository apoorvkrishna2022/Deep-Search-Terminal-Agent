"""
Fallback web searcher implementation.

This is a simpler implementation used as a fallback when the dynamic LangGraph
generation fails. It uses a basic search and summarize approach.
"""

import os
from typing import Dict, Any, List
import json
from rich.console import Console

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from tools.web_search import WebSearchTool
from tools.registry import get_registry

# Initialize console for output
console = Console()

def search_and_summarize(problem: str, context: str = "") -> Dict[str, Any]:
    """
    Search for information related to the problem and summarize the results.
    
    Args:
        problem: The problem to solve.
        context: Optional previous conversation context.
        
    Returns:
        A dictionary with the search results and summary.
    """
    # Initialize tools
    search_tool = WebSearchTool()
    
    # Initialize LLM
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-03-15-preview"),
        temperature=0.1,
        max_tokens=4000,
    )
    
    # Step 1: Analyze the problem and generate search queries
    console.print("[blue]Fallback Mode:[/] Analyzing problem and generating search queries...")
    
    analysis_prompt = """You are an expert at breaking down complex problems into searchable queries.
Given a problem statement and optional context from previous conversations, generate 1-3 specific search queries
that would help find information to solve the problem.

Format your response as a JSON array of search queries.
Example: ["query 1", "query 2", "query 3"]
"""
    
    context_text = f"Previous conversation context:\n{context}\n\n" if context else ""
    
    analysis_messages = [
        SystemMessage(content=analysis_prompt),
        HumanMessage(content=f"{context_text}Problem: {problem}")
    ]
    
    query_response = llm.invoke(analysis_messages)
    
    try:
        # Try to parse as JSON
        queries = json.loads(query_response.content)
        if not isinstance(queries, list):
            queries = [problem]  # Fallback to using the problem as the query
    except:
        # If parsing fails, fallback to using the problem as the query
        queries = [problem]
    
    # Step 2: Perform searches
    all_search_results = []
    
    for i, query in enumerate(queries):
        console.print(f"[blue]Fallback Mode:[/] Searching for '{query}'...")
        results = search_tool.run(query)
        all_search_results.append(f"=== SEARCH RESULTS FOR: {query} ===\n{results}\n")
    
    combined_results = "\n".join(all_search_results)
    
    # Step 3: Synthesize the results
    console.print("[blue]Fallback Mode:[/] Synthesizing results...")
    
    synthesis_prompt = """You are a helpful AI assistant that can answer questions based on search results.
Your task is to analyze the search results and provide a comprehensive, accurate answer to the user's question.
Be clear, informative, and objective in your response. If the search results don't provide enough information,
acknowledge the limitations of your answer.

Structure your answer in a clear, organized way. Use headings if appropriate, and format your response to be easy to read.

If the search results contain contradictory information, acknowledge the different perspectives.
"""
    
    context_text = f"Previous conversation context:\n{context}\n\n" if context else ""
    
    synthesis_messages = [
        SystemMessage(content=synthesis_prompt),
        HumanMessage(content=f"""
{context_text}
Question: {problem}

Search Results:
{combined_results}

Please provide a comprehensive answer based on these search results.
""")
    ]
    
    response = llm.invoke(synthesis_messages)
    
    return {
        "answer": response.content,
        "search_results": combined_results,
        "queries_used": queries
    }

def solve(problem: str, context: str = "") -> Dict[str, Any]:
    """
    Main solve function used as a fallback when dynamic graph generation fails.
    
    Args:
        problem: The problem to solve.
        context: Optional previous conversation context.
        
    Returns:
        The solution to the problem.
    """
    result = search_and_summarize(problem, context)
    return result 