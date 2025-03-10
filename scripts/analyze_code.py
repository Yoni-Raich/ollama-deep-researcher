#!/usr/bin/env python
"""
Code Analysis Runner

This script runs the Code Analysis System on a specified project directory.
It uses the LangGraph-based decision graph to dynamically analyze a codebase
and provide intelligent insights based on a user query.
"""

import os
import sys
import argparse
from pprint import pprint
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.assistant import (
    code_analysis_graph,
    CodeAnalysisStateInput,
    Configuration,
    LLMProvider
)

def main():
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the Code Analysis System on a codebase")
    parser.add_argument("--query", type=str, required=True, 
                        help="The analysis query to execute")
    parser.add_argument("--project-path", type=str, required=True,
                        help="Path to the project to analyze")
    parser.add_argument("--max-analysis-loops", type=int, default=5,
                        help="Maximum number of analysis loops to perform")
    parser.add_argument("--llm-provider", type=str, choices=["ollama", "azure_openai"],
                        default=os.getenv("LLM_PROVIDER", "ollama"),
                        help="LLM provider to use")
    parser.add_argument("--ollama-model", type=str, 
                        default=os.getenv("OLLAMA_MODEL", "llama3.2"),
                        help="Ollama model to use if LLM provider is 'ollama'")
    parser.add_argument("--ollama-base-url", type=str,
                        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/"),
                        help="Ollama API base URL")
    parser.add_argument("--vector-db-path", type=str,
                        default=os.getenv("VECTOR_DB_PATH", "./vector_db"),
                        help="Path to store the vector database")
    
    args = parser.parse_args()
    
    # Create configuration for the analysis run
    config = {
        "configurable": {
            "project_path": args.project_path,
            "max_analysis_loops": args.max_analysis_loops,
            "llm_provider": LLMProvider(args.llm_provider),
            "local_llm": args.ollama_model,
            "ollama_base_url": args.ollama_base_url,
            "vector_db_path": args.vector_db_path,
            "enable_semantic_search": True,
            "enable_reference_analysis": True,
            "enable_dependency_analysis": True,
            "enable_file_structure_analysis": True,
            "enable_signature_analysis": True,
            "enable_content_analysis": True,
        }
    }
    
    # Prepare the input state
    input_state = CodeAnalysisStateInput(
        original_query=args.query,
        project_path=args.project_path
    )
    
    print(f"Starting code analysis for query: {args.query}")
    print(f"Project path: {args.project_path}")
    print(f"LLM provider: {args.llm_provider}")
    print("Running analysis...")
    
    # Execute the analysis graph
    try:
        result = code_analysis_graph.invoke(input_state, config=config)
        
        # Print the result summary
        print("\n==== Analysis Summary ====\n")
        print(result.analysis_summary)
        
        print("\n==== Relevant Files ====\n")
        for file in sorted(result.relevant_files):
            print(f"- {file}")
        
        print("\n==== Analysis Complete ====\n")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 