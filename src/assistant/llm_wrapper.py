import os
import json
from typing import List, Dict, Any, Optional, Union
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception

import httpx
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from assistant.configuration import Configuration, LLMProvider


# Define a helper to detect rate limit errors
def is_rate_limit_error(exception):
    return isinstance(exception, httpx.HTTPStatusError) and exception.response.status_code == 429


# Create a subclass that adds retry logic to embeddings calls
class RetryAzureOpenAIEmbeddings(AzureOpenAIEmbeddings):
    @retry(wait=wait_exponential(multiplier=1, min=10, max=60),
           stop=stop_after_attempt(5),
           retry=retry_if_exception(is_rate_limit_error))
    def embed_documents(self, texts):
        return super().embed_documents(texts)

    @retry(wait=wait_exponential(multiplier=1, min=10, max=60),
           stop=stop_after_attempt(5),
           retry=retry_if_exception(is_rate_limit_error))
    def embed_query(self, text):
        return super().embed_query(text)


class LLMWrapper:
    """
    A wrapper class that provides a unified interface for interacting with 
    different LLM providers (Ollama and Azure OpenAI).
    """
    
    def __init__(self, config: Configuration):
        """
        Initialize the LLM wrapper with the provided configuration.
        
        Args:
            config: The Configuration object containing LLM settings
        """
        self.config = config
        self.provider = config.llm_provider
        
        # Initialize the appropriate LLM based on the provider
        if self.provider == LLMProvider.OLLAMA.value:
            self.llm = ChatOllama(
                base_url=config.ollama_base_url,
                model=config.local_llm,
                temperature=0
            )
            self.llm_json = ChatOllama(
                base_url=config.ollama_base_url,
                model=config.local_llm,
                temperature=0,
                format="json"
            )
            self.embeddings = None  # Will be initialized on-demand
            
        elif self.provider == LLMProvider.AZURE_OPENAI.value:
            self.llm = AzureChatOpenAI(
                azure_endpoint=config.azure_openai_endpoint,
                api_key=config.azure_openai_api_key,
                api_version=config.azure_openai_api_version,
                deployment_name=config.azure_openai_deployment,
                temperature=0
            )
            self.llm_json = AzureChatOpenAI(
                azure_endpoint=config.azure_openai_endpoint,
                api_key=config.azure_openai_api_key,
                api_version=config.azure_openai_api_version,
                deployment_name=config.azure_openai_deployment,
                temperature=0,
                response_format={"type": "json_object"}
            )
            self.embeddings = RetryAzureOpenAIEmbeddings(
                azure_endpoint=config.azure_openai_endpoint,
                api_key=config.azure_openai_api_key,
                api_version=config.azure_openai_api_version,
                deployment_name=config.azure_openai_embedding_deployment
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def get_embeddings(self):
        """
        Get the embeddings model for semantic search.
        If using Ollama and embeddings not initialized, create a basic wrapper.
        
        Returns:
            The embeddings model
        """
        if self.provider == LLMProvider.OLLAMA and self.embeddings is None:
            # For Ollama, we'll use a simple wrapper for the embeddings
            # This assumes Ollama supports the embeddings API
            from langchain_community.embeddings import OllamaEmbeddings
            self.embeddings = OllamaEmbeddings(
                base_url=self.config.ollama_base_url,
                model=self.config.local_llm
            )
        return self.embeddings
    
    def invoke(self, messages: List[BaseMessage]) -> str:
        """
        Invoke the LLM with the provided messages.
        
        Args:
            messages: A list of messages to send to the LLM
            
        Returns:
            The LLM's response as a string
        """
        response = self.llm.invoke(messages)
        return response.content
    
    def invoke_json(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """
        Invoke the LLM with the provided messages and parse the response as JSON.
        
        Args:
            messages: A list of messages to send to the LLM
            
        Returns:
            The LLM's response parsed as a dictionary
        """
        response = self.llm_json.invoke(messages)
        
        try:
            if isinstance(response.content, dict):
                return response.content
            else:
                return json.loads(response.content)
        except json.JSONDecodeError:
            # If JSON parsing fails, extract JSON from the content
            content = response.content
            try:
                # Try to extract JSON object if it's wrapped in text
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    return json.loads(json_str)
                else:
                    # Return a default error response
                    return {"error": "Failed to parse JSON from response", "raw_content": content}
            except Exception:
                # Return a default error response
                return {"error": "Failed to parse JSON from response", "raw_content": content} 