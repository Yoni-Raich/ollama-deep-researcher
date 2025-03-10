# Code Analysis System Guide

This guide explains how to use the Code Analysis System to explore and understand complex codebases.

## Overview

The Code Analysis System is an AI-powered tool that analyzes codebases through multiple analysis methods. It uses a dynamic decision graph to select the most appropriate analysis approaches based on your query and the structure of the codebase.

## Getting Started

### Prerequisites

1. Ensure you have Ollama running locally or have access to Azure OpenAI
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

To analyze a codebase, use the `analyze_code.py` script:

```bash
python scripts/analyze_code.py --query "Your question about the code" --project-path /path/to/your/project
```

For example:

```bash
python scripts/analyze_code.py --query "How is user authentication implemented?" --project-path ./my-web-application
```

### Sample Queries

Here are some effective types of queries you can use:

1. **Architecture Understanding**:
   - "What is the overall architecture of this codebase?"
   - "How are components organized in this project?"
   - "What design patterns are used in this codebase?"

2. **Feature Tracing**:
   - "How is user authentication implemented?"
   - "What is the data flow for processing user uploads?"
   - "How does the payment processing feature work?"

3. **Dependency Mapping**:
   - "What are the dependencies of the AuthService?"
   - "Which components depend on the database module?"
   - "Map the relationships between core modules"

4. **Bug Investigation**:
   - "Why might the login process fail when using OAuth?"
   - "What could cause the memory leak in the file processing module?"
   - "Why would the API return 500 errors when processing large requests?"

5. **Security Analysis**:
   - "Are there any potential SQL injection vulnerabilities?"
   - "How is input validation handled for user-supplied data?"
   - "What security measures are in place for API endpoints?"

## Advanced Configuration

### LLM Configuration

You can configure which LLM to use (Ollama or Azure OpenAI) by setting environment variables or command-line arguments:

For Ollama:
```bash
python scripts/analyze_code.py --llm-provider ollama --ollama-model llama3.2 --query "Your query" --project-path ./project
```

For Azure OpenAI:
```bash
# Set these in your .env file:
# AZURE_OPENAI_ENDPOINT=your-endpoint
# AZURE_OPENAI_API_KEY=your-api-key
# AZURE_OPENAI_DEPLOYMENT=your-deployment
# AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your-embedding-deployment

python scripts/analyze_code.py --llm-provider azure_openai --query "Your query" --project-path ./project
```

### Analysis Method Customization

You can enable or disable specific analysis methods by editing the `.env` file:

```
# Analysis method settings
ENABLE_SEMANTIC_SEARCH=True
ENABLE_REFERENCE_ANALYSIS=True
ENABLE_DEPENDENCY_ANALYSIS=True
ENABLE_FILE_STRUCTURE_ANALYSIS=True
ENABLE_SIGNATURE_ANALYSIS=True
ENABLE_CONTENT_ANALYSIS=True
```

### External Tool Integration

For more precise code analysis, you can enable support for external tools:

```
# Tool-specific settings
USE_TREE_SITTER=True  # For precise code parsing
USE_CTAGS=True        # For better reference analysis
```

Note: These tools need to be installed separately on your system.

## Understanding the Analysis Process

The analysis system follows this general workflow:

1. **Task Analysis**: Categorizes your query and determines its complexity
2. **Strategy Selection**: Chooses the most appropriate analysis methods
3. **Analysis Execution**: Runs the selected analysis methods in the optimal order
4. **Result Integration**: Combines insights from different analysis methods
5. **Knowledge Gap Identification**: Identifies what is still unknown
6. **Iterative Analysis**: Repeats analysis with refined approaches
7. **Final Summary**: Produces a comprehensive report

## Interpreting Results

The final output provides:

1. **Analysis Summary**: A comprehensive answer to your query
2. **Relevant Files**: Files identified as most relevant to your query
3. **Knowledge Gathered**: Structured information about the codebase
4. **Analysis History**: The steps taken during the analysis

## Troubleshooting

### Common Issues

1. **Limited or incorrect results**:
   - Try refining your query to be more specific
   - Increase the `MAX_ANALYSIS_LOOPS` setting (default is 5)
   - Enable additional analysis methods that may be relevant

2. **Performance issues**:
   - Limit analysis to specific subdirectories if the project is large
   - Adjust the `MAX_FILES_TO_ANALYZE` setting to reduce the scope
   - Reduce the `MAX_DEPTH` setting for dependency analysis

3. **Tool integration issues**:
   - Ensure tree-sitter or ctags are correctly installed if enabled
   - Check the logs for specific errors related to external tools

## Best Practices

1. **Start broad, then narrow down**: Begin with high-level queries about architecture, then drill down into specific components.

2. **Use specific terminology**: Include specific class names, method names, or keywords from the codebase when you know them.

3. **Analyze iteratively**: Use insights from one analysis to inform more specific questions in subsequent analyses.

4. **Combine with manual exploration**: Use the analysis results as a guide for more detailed manual code review. 