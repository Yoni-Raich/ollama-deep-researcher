import json
from typing import Dict, Any, List, Optional
from typing_extensions import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph

from assistant.configuration import Configuration
from assistant.state import CodeAnalysisState, CodeAnalysisStateInput, CodeAnalysisStateOutput
from assistant.code_search import CodeSearchWrapper
from assistant.llm_wrapper import LLMWrapper
from assistant.prompts import (
    task_analysis_prompt,
    strategy_selection_prompt,
    semantic_search_prompt,
    reference_analysis_prompt,
    dependency_analysis_prompt,
    file_structure_analysis_prompt,
    signature_analysis_prompt,
    content_analysis_prompt,
    result_integration_prompt,
    knowledge_gap_prompt,
    finalization_prompt
)

# Initialize global objects
def initialize_objects(config: Configuration) -> Dict[str, Any]:
    """Initialize the objects needed for code analysis."""
    llm_wrapper = LLMWrapper(config)
    code_search = CodeSearchWrapper(config, llm_wrapper)
    return {
        "llm_wrapper": llm_wrapper,
        "code_search": code_search
    }

# Nodes
def task_analysis(state: CodeAnalysisState, config: RunnableConfig) -> Dict[str, Any]:
    """Analyze the task to determine appropriate analysis strategies."""
    configurable = Configuration.from_runnable_config(config)
    objects = initialize_objects(configurable)
    llm_wrapper = objects["llm_wrapper"]
    
    # Format the task analysis prompt
    task_prompt = task_analysis_prompt.format(original_query=state.original_query)
    
    # Use JSON mode for structured output
    result = llm_wrapper.invoke_json([
        SystemMessage(content=task_prompt),
        HumanMessage(content=f"Analyze the following code research task: {state.original_query}")
    ])
    
    # Update state with task analysis results
    task_category = result.get("task_category", "unknown")
    task_complexity = result.get("task_complexity", "medium")
    current_query = result.get("refined_query", state.original_query)
    
    # Record the analysis in history
    analysis_entry = {
        "step": "task_analysis",
        "task_category": task_category,
        "task_complexity": task_complexity,
        "refined_query": current_query
    }
    
    return {
        "task_category": task_category,
        "task_complexity": task_complexity,
        "current_query": current_query,
        "analysis_history": [analysis_entry]
    }

def strategy_selection(state: CodeAnalysisState, config: RunnableConfig) -> Dict[str, Any]:
    """Select appropriate strategies based on task analysis."""
    configurable = Configuration.from_runnable_config(config)
    objects = initialize_objects(configurable)
    llm_wrapper = objects["llm_wrapper"]
    
    # Format the strategy selection prompt
    strategy_prompt = strategy_selection_prompt.format(
        original_query=state.original_query,
        task_category=state.task_category,
        task_complexity=state.task_complexity
    )
    
    # Use JSON mode for structured output
    result = llm_wrapper.invoke_json([
        SystemMessage(content=strategy_prompt),
        HumanMessage(content=f"Select appropriate analysis strategies for: {state.current_query}")
    ])
    
    # Extract the strategies to use
    strategies = result.get("selected_strategies", ["semantic_search"])
    priority_order = result.get("priority_order", [])
    parallel_execution = result.get("parallel_execution", [])
    
    # Record the strategy selection in history
    strategy_entry = {
        "step": "strategy_selection",
        "selected_strategies": strategies,
        "priority_order": priority_order,
        "parallel_execution": parallel_execution
    }
    
    return {
        "current_approach": strategies[0] if strategies else "semantic_search",
        "analysis_history": [strategy_entry]
    }

def semantic_search(state: CodeAnalysisState, config: RunnableConfig) -> Dict[str, Any]:
    """Perform semantic search on the codebase."""
    configurable = Configuration.from_runnable_config(config)
    objects = initialize_objects(configurable)
    llm_wrapper = objects["llm_wrapper"]
    code_search = objects["code_search"]
    
    # Skip if semantic search is not enabled
    if not configurable.enable_semantic_search:
        return {"analysis_results": {**state.analysis_results, "semantic_search": {"skipped": True}}}
    
    # Perform semantic search
    search_results = code_search.semantic_code_search(
        query=state.current_query,
        k=10,
        exclude_files=list(state.relevant_files) if state.relevant_files else None
    )
    
    # Process the results with LLM to extract insights
    search_prompt = semantic_search_prompt.format(
        original_query=state.original_query,
        current_query=state.current_query,
        search_results=json.dumps(search_results, indent=2)
    )
    
    result = llm_wrapper.invoke_json([
        SystemMessage(content=search_prompt),
        HumanMessage(content="Extract insights from these search results.")
    ])
    
    # Extract relevant files
    relevant_files = set(result.get("relevant_files", []))
    
    # Update the analysis results
    analysis_entry = {
        "step": "semantic_search",
        "query": state.current_query,
        "raw_results": search_results,
        "insights": result.get("insights", {}),
        "relevant_files": list(relevant_files)
    }
    
    return {
        "analysis_results": {**state.analysis_results, "semantic_search": result.get("insights", {})},
        "relevant_files": relevant_files,
        "analysis_history": [analysis_entry]
    }

def reference_analysis(state: CodeAnalysisState, config: RunnableConfig) -> Dict[str, Any]:
    """Perform reference analysis on the codebase."""
    configurable = Configuration.from_runnable_config(config)
    objects = initialize_objects(configurable)
    llm_wrapper = objects["llm_wrapper"]
    code_search = objects["code_search"]
    
    # Skip if reference analysis is not enabled
    if not configurable.enable_reference_analysis:
        return {"analysis_results": {**state.analysis_results, "reference_analysis": {"skipped": True}}}
    
    # Extract entities to analyze from the current knowledge
    analysis_prompt = reference_analysis_prompt.format(
        original_query=state.original_query,
        current_query=state.current_query,
        knowledge_gathered=json.dumps(state.knowledge_gathered, indent=2)
    )
    
    entity_result = llm_wrapper.invoke_json([
        SystemMessage(content=analysis_prompt),
        HumanMessage(content="Identify key entities for reference analysis.")
    ])
    
    entities = entity_result.get("entities", [])
    if not entities and state.relevant_files:
        # If no entities extracted but we have relevant files, use filename-based entities
        entities = [path.split("/")[-1].split(".")[0] for path in state.relevant_files]
    
    # Perform reference analysis for each entity
    reference_results = {}
    for entity in entities[:5]:  # Limit to prevent too many searches
        refs = code_search.reference_analysis(entity)
        if refs:
            reference_results[entity] = refs
    
    # Process the results with LLM
    if reference_results:
        result = llm_wrapper.invoke_json([
            SystemMessage(content=reference_analysis_prompt),
            HumanMessage(content=f"Extract insights from these reference results: {json.dumps(reference_results, indent=2)}")
        ])
        
        # Extract new relevant files
        new_relevant_files = set(result.get("relevant_files", []))
        
        # Update the analysis results
        analysis_entry = {
            "step": "reference_analysis",
            "entities": entities,
            "raw_results": reference_results,
            "insights": result.get("insights", {}),
            "relevant_files": list(new_relevant_files)
        }
        
        return {
            "analysis_results": {**state.analysis_results, "reference_analysis": result.get("insights", {})},
            "relevant_files": state.relevant_files.union(new_relevant_files),
            "analysis_history": [analysis_entry]
        }
    
    return {"analysis_results": {**state.analysis_results, "reference_analysis": {"no_results": True}}}

def dependency_analysis(state: CodeAnalysisState, config: RunnableConfig) -> Dict[str, Any]:
    """Analyze dependencies between files and modules."""
    configurable = Configuration.from_runnable_config(config)
    objects = initialize_objects(configurable)
    llm_wrapper = objects["llm_wrapper"]
    code_search = objects["code_search"]
    
    # Skip if dependency analysis is not enabled
    if not configurable.enable_dependency_analysis:
        return {"analysis_results": {**state.analysis_results, "dependency_analysis": {"skipped": True}}}
    
    # Select files for dependency analysis
    files_to_analyze = list(state.relevant_files)[:configurable.max_files_to_analyze]
    
    # Perform dependency analysis
    dependency_results = {}
    for file_path in files_to_analyze:
        deps = code_search.dependency_analysis(file_path)
        if deps:
            dependency_results[file_path] = deps
    
    # Process the results with LLM
    if dependency_results:
        result = llm_wrapper.invoke_json([
            SystemMessage(content=dependency_analysis_prompt),
            HumanMessage(content=f"Extract insights from these dependency results: {json.dumps(dependency_results, indent=2)}")
        ])
        
        # Update the knowledge gathered with dependency information
        knowledge = state.knowledge_gathered.copy()
        knowledge["dependencies"] = knowledge.get("dependencies", {})
        knowledge["dependencies"].update(result.get("dependencies", {}))
        
        # Update the analysis results
        analysis_entry = {
            "step": "dependency_analysis",
            "files_analyzed": files_to_analyze,
            "raw_results": dependency_results,
            "insights": result.get("insights", {})
        }
        
        return {
            "analysis_results": {**state.analysis_results, "dependency_analysis": result.get("insights", {})},
            "knowledge_gathered": knowledge,
            "analysis_history": [analysis_entry]
        }
    
    return {"analysis_results": {**state.analysis_results, "dependency_analysis": {"no_results": True}}}

def file_structure_analysis(state: CodeAnalysisState, config: RunnableConfig) -> Dict[str, Any]:
    """Analyze the file structure of the codebase."""
    configurable = Configuration.from_runnable_config(config)
    objects = initialize_objects(configurable)
    llm_wrapper = objects["llm_wrapper"]
    code_search = objects["code_search"]
    
    # Skip if file structure analysis is not enabled
    if not configurable.enable_file_structure_analysis:
        return {"analysis_results": {**state.analysis_results, "file_structure_analysis": {"skipped": True}}}
    
    # Perform file structure analysis
    structure_results = code_search.file_structure_analysis()
    
    # Process the results with LLM
    if structure_results:
        result = llm_wrapper.invoke_json([
            SystemMessage(content=file_structure_analysis_prompt),
            HumanMessage(content=f"Extract insights from this file structure: {json.dumps(structure_results, indent=2)}")
        ])
        
        # Update the knowledge gathered with structure information
        knowledge = state.knowledge_gathered.copy()
        knowledge["file_structure"] = result.get("structure_insights", {})
        
        # Update the analysis results
        analysis_entry = {
            "step": "file_structure_analysis",
            "raw_results": structure_results,
            "insights": result.get("insights", {})
        }
        
        return {
            "analysis_results": {**state.analysis_results, "file_structure_analysis": result.get("insights", {})},
            "knowledge_gathered": knowledge,
            "analysis_history": [analysis_entry]
        }
    
    return {"analysis_results": {**state.analysis_results, "file_structure_analysis": {"no_results": True}}}

def signature_analysis(state: CodeAnalysisState, config: RunnableConfig) -> Dict[str, Any]:
    """Analyze function and class signatures in the codebase."""
    configurable = Configuration.from_runnable_config(config)
    objects = initialize_objects(configurable)
    llm_wrapper = objects["llm_wrapper"]
    code_search = objects["code_search"]
    
    # Skip if signature analysis is not enabled
    if not configurable.enable_signature_analysis:
        return {"analysis_results": {**state.analysis_results, "signature_analysis": {"skipped": True}}}
    
    # Select files for signature analysis
    files_to_analyze = list(state.relevant_files)[:configurable.max_files_to_analyze]
    
    # Perform signature analysis
    signature_results = {}
    for file_path in files_to_analyze:
        sigs = code_search.signature_analysis(file_path)
        if sigs:
            signature_results[file_path] = sigs
    
    # Process the results with LLM
    if signature_results:
        result = llm_wrapper.invoke_json([
            SystemMessage(content=signature_analysis_prompt),
            HumanMessage(content=f"Extract insights from these signature results: {json.dumps(signature_results, indent=2)}")
        ])
        
        # Update the knowledge gathered with signature information
        knowledge = state.knowledge_gathered.copy()
        knowledge["signatures"] = knowledge.get("signatures", {})
        knowledge["signatures"].update(result.get("signature_insights", {}))
        
        # Update the analysis results
        analysis_entry = {
            "step": "signature_analysis",
            "files_analyzed": files_to_analyze,
            "raw_results": signature_results,
            "insights": result.get("insights", {})
        }
        
        return {
            "analysis_results": {**state.analysis_results, "signature_analysis": result.get("insights", {})},
            "knowledge_gathered": knowledge,
            "analysis_history": [analysis_entry]
        }
    
    return {"analysis_results": {**state.analysis_results, "signature_analysis": {"no_results": True}}}

def content_analysis(state: CodeAnalysisState, config: RunnableConfig) -> Dict[str, Any]:
    """Search for specific content patterns in the codebase."""
    configurable = Configuration.from_runnable_config(config)
    objects = initialize_objects(configurable)
    llm_wrapper = objects["llm_wrapper"]
    code_search = objects["code_search"]
    
    # Skip if content analysis is not enabled
    if not configurable.enable_content_analysis:
        return {"analysis_results": {**state.analysis_results, "content_analysis": {"skipped": True}}}
    
    # Extract patterns to search for
    analysis_prompt = content_analysis_prompt.format(
        original_query=state.original_query,
        current_query=state.current_query,
        knowledge_gathered=json.dumps(state.knowledge_gathered, indent=2)
    )
    
    pattern_result = llm_wrapper.invoke_json([
        SystemMessage(content=analysis_prompt),
        HumanMessage(content="Identify key patterns for content analysis.")
    ])
    
    patterns = pattern_result.get("patterns", [])
    if not patterns:
        # If no patterns extracted, use current query keywords
        patterns = [state.current_query]
    
    # Perform content analysis for each pattern
    content_results = {}
    for pattern in patterns[:3]:  # Limit to prevent too many searches
        results = code_search.content_analysis(pattern)
        if results:
            content_results[pattern] = results
    
    # Process the results with LLM
    if content_results:
        result = llm_wrapper.invoke_json([
            SystemMessage(content=content_analysis_prompt),
            HumanMessage(content=f"Extract insights from these content results: {json.dumps(content_results, indent=2)}")
        ])
        
        # Extract new relevant files
        new_relevant_files = set(result.get("relevant_files", []))
        
        # Update the analysis results
        analysis_entry = {
            "step": "content_analysis",
            "patterns": patterns,
            "raw_results": content_results,
            "insights": result.get("insights", {}),
            "relevant_files": list(new_relevant_files)
        }
        
        return {
            "analysis_results": {**state.analysis_results, "content_analysis": result.get("insights", {})},
            "relevant_files": state.relevant_files.union(new_relevant_files),
            "analysis_history": [analysis_entry]
        }
    
    return {"analysis_results": {**state.analysis_results, "content_analysis": {"no_results": True}}}

def result_integration(state: CodeAnalysisState, config: RunnableConfig) -> Dict[str, Any]:
    """Integrate results from different analysis methods."""
    configurable = Configuration.from_runnable_config(config)
    objects = initialize_objects(configurable)
    llm_wrapper = objects["llm_wrapper"]
    
    # Format the integration prompt
    integration_prompt = result_integration_prompt.format(
        original_query=state.original_query,
        current_query=state.current_query,
        analysis_results=json.dumps(state.analysis_results, indent=2),
        relevant_files=json.dumps(list(state.relevant_files), indent=2)
    )
    
    # Use the LLM to integrate results
    result = llm_wrapper.invoke_json([
        SystemMessage(content=integration_prompt),
        HumanMessage(content="Integrate the analysis results to form a coherent understanding.")
    ])
    
    # Update the knowledge with integrated results
    knowledge = state.knowledge_gathered.copy()
    
    # Add new insights from integration
    for key, value in result.get("integrated_knowledge", {}).items():
        if key in knowledge:
            if isinstance(knowledge[key], dict) and isinstance(value, dict):
                knowledge[key].update(value)
            elif isinstance(knowledge[key], list) and isinstance(value, list):
                knowledge[key].extend(value)
            else:
                knowledge[key] = value
        else:
            knowledge[key] = value
    
    # Record the integration in history
    integration_entry = {
        "step": "result_integration",
        "integrated_knowledge": result.get("integrated_knowledge", {}),
        "key_findings": result.get("key_findings", [])
    }
    
    return {
        "knowledge_gathered": knowledge,
        "analysis_history": [integration_entry]
    }

def knowledge_gap_identification(state: CodeAnalysisState, config: RunnableConfig) -> Dict[str, Any]:
    """Identify knowledge gaps and determine next steps."""
    configurable = Configuration.from_runnable_config(config)
    objects = initialize_objects(configurable)
    llm_wrapper = objects["llm_wrapper"]
    
    # Format the knowledge gap prompt
    gap_prompt = knowledge_gap_prompt.format(
        original_query=state.original_query,
        current_query=state.current_query,
        knowledge_gathered=json.dumps(state.knowledge_gathered, indent=2),
        analysis_history=json.dumps(state.analysis_history, indent=2)
    )
    
    # Use the LLM to identify knowledge gaps
    result = llm_wrapper.invoke_json([
        SystemMessage(content=gap_prompt),
        HumanMessage(content="Identify knowledge gaps and determine next steps.")
    ])
    
    # Extract knowledge gaps and next approach
    knowledge_gaps = result.get("knowledge_gaps", [])
    next_approach = result.get("next_approach")
    refined_query = result.get("refined_query")
    
    # Record the gap identification in history
    gap_entry = {
        "step": "knowledge_gap_identification",
        "knowledge_gaps": knowledge_gaps,
        "next_approach": next_approach,
        "refined_query": refined_query
    }
    
    return {
        "knowledge_gaps": knowledge_gaps,
        "current_approach": next_approach if next_approach else state.current_approach,
        "current_query": refined_query if refined_query else state.current_query,
        "analysis_history": [gap_entry],
        "analysis_loop_count": state.analysis_loop_count + 1
    }

def finalize_analysis(state: CodeAnalysisState, config: RunnableConfig) -> Dict[str, Any]:
    """Finalize the analysis and create a comprehensive summary."""
    configurable = Configuration.from_runnable_config(config)
    objects = initialize_objects(configurable)
    llm_wrapper = objects["llm_wrapper"]
    
    # Format the finalization prompt
    finalization_prompt_text = finalization_prompt.format(
        original_query=state.original_query,
        knowledge_gathered=json.dumps(state.knowledge_gathered, indent=2),
        relevant_files=json.dumps(list(state.relevant_files), indent=2),
        analysis_history=json.dumps(state.analysis_history, indent=2)
    )
    
    # Use the LLM to create the final summary
    result = llm_wrapper.invoke([
        SystemMessage(content=finalization_prompt_text),
        HumanMessage(content="Generate a comprehensive analysis summary.")
    ])
    
    # Record the finalization in history
    finalize_entry = {
        "step": "finalize_analysis",
        "summary_generated": True
    }
    
    return {
        "analysis_summary": result,
        "analysis_history": [finalize_entry]
    }

def route_to_next_analysis(state: CodeAnalysisState, config: RunnableConfig) -> str:
    """Route to the appropriate next analysis method based on current approach."""
    # If we've reached the maximum analysis loop count, finalize
    configurable = Configuration.from_runnable_config(config)
    if state.analysis_loop_count >= configurable.max_analysis_loops:
        return "finalize_analysis"
    
    # Otherwise, route based on the current approach
    approach = state.current_approach.lower() if state.current_approach else "semantic_search"
    
    if approach == "semantic_search":
        return "semantic_search"
    elif approach == "reference_analysis":
        return "reference_analysis"
    elif approach == "dependency_analysis":
        return "dependency_analysis"
    elif approach == "file_structure_analysis":
        return "file_structure_analysis"
    elif approach == "signature_analysis":
        return "signature_analysis"
    elif approach == "content_analysis":
        return "content_analysis"
    else:
        # Default to semantic search if approach not recognized
        return "semantic_search"

# Build the graph
builder = StateGraph(CodeAnalysisState, input=CodeAnalysisStateInput, output=CodeAnalysisStateOutput, config_schema=Configuration)

# Add nodes
builder.add_node("task_analysis", task_analysis)
builder.add_node("strategy_selection", strategy_selection)
builder.add_node("semantic_search", semantic_search)
builder.add_node("reference_analysis", reference_analysis)
builder.add_node("dependency_analysis", dependency_analysis)
builder.add_node("file_structure_analysis", file_structure_analysis)
builder.add_node("signature_analysis", signature_analysis)
builder.add_node("content_analysis", content_analysis)
builder.add_node("result_integration", result_integration)
builder.add_node("knowledge_gap_identification", knowledge_gap_identification)
builder.add_node("finalize_analysis", finalize_analysis)

# Add edges
builder.add_edge(START, "task_analysis")
builder.add_edge("task_analysis", "strategy_selection")

# Connect strategy selection to the appropriate next analysis nodes
builder.add_conditional_edges(
    "strategy_selection",
    route_to_next_analysis,
    {
        "semantic_search": "semantic_search",
        "reference_analysis": "reference_analysis",
        "dependency_analysis": "dependency_analysis",
        "file_structure_analysis": "file_structure_analysis",
        "signature_analysis": "signature_analysis", 
        "content_analysis": "content_analysis"
    }
)

# Connect each analysis method to result integration
builder.add_edge("semantic_search", "result_integration")
builder.add_edge("reference_analysis", "result_integration")
builder.add_edge("dependency_analysis", "result_integration")
builder.add_edge("file_structure_analysis", "result_integration")
builder.add_edge("signature_analysis", "result_integration")
builder.add_edge("content_analysis", "result_integration")

# Connect result integration to knowledge gap identification
builder.add_edge("result_integration", "knowledge_gap_identification")

# Connect knowledge gap identification to either the next analysis or finalization
builder.add_conditional_edges(
    "knowledge_gap_identification",
    route_to_next_analysis,
    {
        "semantic_search": "semantic_search",
        "reference_analysis": "reference_analysis",
        "dependency_analysis": "dependency_analysis",
        "file_structure_analysis": "file_structure_analysis",
        "signature_analysis": "signature_analysis",
        "content_analysis": "content_analysis",
        "finalize_analysis": "finalize_analysis"
    }
)

# Connect finalization to END
builder.add_edge("finalize_analysis", END)

# Compile the graph
graph = builder.compile()