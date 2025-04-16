"""
Base class for all agents in the system.
"""

import os
from typing import Dict, Any, List, Optional
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

class AgentConfig(BaseModel):
    """Configuration for an agent."""
    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of what the agent does")
    system_prompt: str = Field(..., description="System prompt for the agent")
    model_name: str = Field(default_factory=lambda: os.getenv("MODEL_NAME", "gpt-4o"))
    temperature: float = Field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.1")))
    max_tokens: int = Field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "4000")))
    tools: List[Any] = Field(default_factory=list)
    azure_endpoint: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    azure_deployment: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""))
    azure_api_version: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2023-03-15-preview"))

class BaseAgent:
    """Base class for all agents."""
    
    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        """Initialize the agent with given configuration."""
        if config:
            self.config = config
        else:
            # Create default config with provided kwargs
            self.config = AgentConfig(**kwargs)
        
        # Initialize LLM with Azure OpenAI
        self.llm = AzureChatOpenAI(
            azure_endpoint=self.config.azure_endpoint,
            azure_deployment=self.config.azure_deployment,
            openai_api_version=self.config.azure_api_version,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent with the given input data.
        
        Args:
            input_data: Input data for the agent.
            
        Returns:
            Dict containing the agent's response.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _call_llm(self, messages: List[Dict[str, Any]]) -> str:
        """
        Call the language model with the given messages.
        
        Args:
            messages: List of message dictionaries.
            
        Returns:
            The model's response as a string.
        """
        formatted_messages = []
        
        for message in messages:
            if message["type"] == "system":
                formatted_messages.append(SystemMessage(content=message["content"]))
            elif message["type"] == "human":
                formatted_messages.append(HumanMessage(content=message["content"]))
        
        response = self.llm.invoke(formatted_messages)
        return response.content 