"""
Web search tool for retrieving information from the internet.
"""

import os
from typing import Dict, Any, List, Optional
import json
from duckduckgo_search import DDGS

class WebSearchTool:
    """Tool for searching the web using DuckDuckGo."""
    
    def __init__(self, max_results: int = 5):
        """
        Initialize the web search tool.
        
        Args:
            max_results: Maximum number of search results to return.
        """
        self.max_results = max_results
        self.search_client = DDGS()
    
    def run(self, query: str) -> str:
        """
        Search the web for the given query.
        
        Args:
            query: The query string to search for.
            
        Returns:
            A string containing search results.
        """
        # Clean up the query
        clean_query = query.strip()
        
        try:
            # Search using DuckDuckGo
            results = list(self.search_client.text(
                clean_query,
                region="wt-wt",
                safesearch="moderate",
                timelimit=None,
                max_results=self.max_results
            ))
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results):
                formatted_results.append(f"[{i+1}] {result['title']}")
                formatted_results.append(f"    URL: {result['href']}")
                formatted_results.append(f"    {result['body']}")
                formatted_results.append("")
            
            if formatted_results:
                return "\n".join(formatted_results)
            else:
                return "No search results found for the query."
        
        except Exception as e:
            return f"Error performing web search: {str(e)}"

# For testing
if __name__ == "__main__":
    tool = WebSearchTool()
    results = tool.run("What is LangGraph?")
    print(results) 