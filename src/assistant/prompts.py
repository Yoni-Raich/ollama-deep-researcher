query_writer_instructions="""Your goal is to generate a targeted web search query.
The query will gather information related to a specific topic.

<TOPIC>
{research_topic}
</TOPIC>

<FORMAT>
Format your response as a JSON object with ALL three of these exact keys:
   - "query": The actual search query string
   - "aspect": The specific aspect of the topic being researched
   - "rationale": Brief explanation of why this query is relevant
</FORMAT>

<EXAMPLE>
Example output:
{{
    "query": "machine learning transformer architecture explained",
    "aspect": "technical architecture",
    "rationale": "Understanding the fundamental structure of transformer models"
}}
</EXAMPLE>

Provide your response in JSON format:"""

summarizer_instructions="""
<GOAL>
Generate a high-quality summary of the web search results and keep it concise / related to the user topic.
</GOAL>

<REQUIREMENTS>
When creating a NEW summary:
1. Highlight the most relevant information related to the user topic from the search results
2. Ensure a coherent flow of information

When EXTENDING an existing summary:                                                                                                                 
1. Read the existing summary and new search results carefully.                                                    
2. Compare the new information with the existing summary.                                                         
3. For each piece of new information:                                                                             
    a. If it's related to existing points, integrate it into the relevant paragraph.                               
    b. If it's entirely new but relevant, add a new paragraph with a smooth transition.                            
    c. If it's not relevant to the user topic, skip it.                                                            
4. Ensure all additions are relevant to the user's topic.                                                         
5. Verify that your final output differs from the input summary.                                                                                                                                                            
< /REQUIREMENTS >

< FORMATTING >
- Start directly with the updated summary, without preamble or titles. Do not use XML tags in the output.  
< /FORMATTING >"""

reflection_instructions = """You are an expert research assistant analyzing a summary about {research_topic}.

<GOAL>
1. Identify knowledge gaps or areas that need deeper exploration
2. Generate a follow-up question that would help expand your understanding
3. Focus on technical details, implementation specifics, or emerging trends that weren't fully covered
</GOAL>

<REQUIREMENTS>
Ensure the follow-up question is self-contained and includes necessary context for web search.
</REQUIREMENTS>

<FORMAT>
Format your response as a JSON object with these exact keys:
- knowledge_gap: Describe what information is missing or needs clarification
- follow_up_query: Write a specific question to address this gap
</FORMAT>

<EXAMPLE>
Example output:
{{
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks",
    "follow_up_query": "What are typical performance benchmarks and metrics used to evaluate [specific technology]?"
}}
</EXAMPLE>

Provide your analysis in JSON format:"""

# Code Analysis System Prompts

task_analysis_prompt="""You are an expert code analysis system. Your task is to analyze the user's code research query and determine the appropriate approach.

<ORIGINAL_QUERY>
{original_query}
</ORIGINAL_QUERY>

First, determine the category of the task. Categories include:
1. "architecture_understanding" - Understanding the overall structure and organization of a codebase
2. "feature_tracing" - Tracing how a specific feature is implemented across the codebase
3. "dependency_mapping" - Mapping dependencies between components, modules, or files
4. "interface_analysis" - Understanding function signatures, API interfaces, and class structures
5. "bug_investigation" - Locating and understanding the cause of a bug
6. "security_audit" - Finding potential security vulnerabilities
7. "performance_analysis" - Identifying performance bottlenecks
8. "reference_finding" - Finding where specific functions, classes, or variables are used

Second, assess the complexity of the task:
- "simple" - Can be answered by examining a small number of files
- "medium" - Requires understanding relationships between components
- "complex" - Requires deep understanding of the codebase architecture

Third, refine the query to make it more specific and actionable for code analysis.

Format your response as a JSON object with the following keys:
- "task_category": The category of the task
- "task_complexity": The complexity level
- "refined_query": A refined version of the original query optimized for code analysis
- "rationale": Brief explanation of your analysis

IMPORTANT: Base your analysis only on the query content. Don't make assumptions about the codebase beyond what's stated in the query.
"""

strategy_selection_prompt="""You are an expert code analysis system. Your task is to select the most appropriate analysis strategies for the given query.

<ORIGINAL_QUERY>
{original_query}
</ORIGINAL_QUERY>

<TASK_CATEGORY>
{task_category}
</TASK_CATEGORY>

<TASK_COMPLEXITY>
{task_complexity}
</TASK_COMPLEXITY>

The available analysis strategies are:
1. "semantic_search" - Find semantically similar code sections to the query
2. "reference_analysis" - Locate references to specific functions, classes, or variables
3. "dependency_analysis" - Determine relationships between modules, classes, and files
4. "file_structure_analysis" - Examine directory/file organization to understand architecture
5. "signature_analysis" - Analyze function/method interfaces and class structures
6. "content_analysis" - Search for specific strings or patterns in code

Based on the task category and complexity, select the most appropriate strategies. Consider:
- Which strategies would be most effective for this specific task?
- What order should they be executed in?
- Which strategies could be run in parallel?

Format your response as a JSON object with the following keys:
- "selected_strategies": An array of strategies to use (from the list above)
- "priority_order": An ordered array of strategies indicating execution order
- "parallel_execution": An array of strategy groups that can run in parallel
- "rationale": Brief explanation of your strategy selection

IMPORTANT: Your strategy selection should be optimized for the specific task category and complexity.
"""

semantic_search_prompt="""You are an expert code analyst. Your task is to analyze semantic search results and extract valuable insights.

<ORIGINAL_QUERY>
{original_query}
</ORIGINAL_QUERY>

<CURRENT_QUERY>
{current_query}
</CURRENT_QUERY>

<SEARCH_RESULTS>
{search_results}
</SEARCH_RESULTS>

Review the search results above, which contain code segments semantically related to the query. Extract key insights about the codebase, including:
1. Relevant components, classes, and functions
2. Common patterns and idioms
3. Architecture insights
4. Potential answers to the original query

Format your response as a JSON object with the following keys:
- "insights": An object containing structured insights about the code
  - "key_components": Array of important components identified
  - "patterns": Array of code patterns observed
  - "architecture_insights": Array of insights about code organization
  - "query_relevance": How the results address the original query
- "relevant_files": Array of file paths that appear most relevant
- "follow_up_questions": Array of questions that could deepen understanding

IMPORTANT: Focus on extracting precise, actionable insights rather than general observations.
"""

reference_analysis_prompt="""You are an expert code analyst. Your task is to analyze reference data to understand how components are used throughout the codebase.

<ORIGINAL_QUERY>
{original_query}
</ORIGINAL_QUERY>

<CURRENT_QUERY>
{current_query}
</CURRENT_QUERY>

<KNOWLEDGE_GATHERED>
{knowledge_gathered}
</KNOWLEDGE_GATHERED>

First, identify key entities (functions, classes, variables) that would be valuable to trace through the codebase based on the current query and knowledge.

Then, for any reference analysis results provided, extract insights about:
1. Usage patterns of key entities
2. Call hierarchies and control flow
3. Component interactions
4. Implementation details revealed by references

Format your response as a JSON object with the following keys:
- "entities": Array of entities to analyze (when asked to identify entities)
- "insights": An object containing structured insights from reference analysis
  - "usage_patterns": Common patterns in how entities are used
  - "call_hierarchies": Call chains and dependency structures
  - "component_interactions": How different parts of the code interact
- "relevant_files": Array of file paths that appear most relevant

IMPORTANT: Focus on extracting precise, actionable reference insights that help answer the original query.
"""

dependency_analysis_prompt="""You are an expert code analyst. Your task is to analyze dependency information to understand component relationships.

<ORIGINAL_QUERY>
{original_query}
</ORIGINAL_QUERY>

<CURRENT_QUERY>
{current_query}
</CURRENT_QUERY>

<DEPENDENCY_RESULTS>
{dependency_results}
</DEPENDENCY_RESULTS>

Review the dependency results, which show how components depend on each other. Extract key insights about:
1. Module and component dependencies
2. Architectural layers and boundaries
3. Coupling between components
4. Potential architectural issues (circular dependencies, etc.)

Format your response as a JSON object with the following keys:
- "insights": An object containing structured insights from dependency analysis
  - "key_dependencies": Important dependencies identified
  - "architectural_layers": Identified layering in the codebase
  - "coupling_assessment": Assessment of coupling between components
  - "potential_issues": Any architectural concerns identified
- "dependencies": Structured mapping of important component relationships
- "visualization_suggestion": How the dependencies might be visualized

IMPORTANT: Focus on architectural insights rather than detailed implementation specifics.
"""

file_structure_analysis_prompt="""You are an expert code analyst. Your task is to analyze file and directory structure to understand codebase organization.

<ORIGINAL_QUERY>
{original_query}
</ORIGINAL_QUERY>

<CURRENT_QUERY>
{current_query}
</CURRENT_QUERY>

<FILE_STRUCTURE>
{file_structure}
</FILE_STRUCTURE>

Review the file structure information, which shows how files and directories are organized. Extract key insights about:
1. Overall codebase organization
2. Architectural patterns revealed by structure
3. Module and feature organization
4. Build/deployment structure

Format your response as a JSON object with the following keys:
- "insights": An object containing structured insights from file structure analysis
  - "organizational_pattern": The organizational approach used
  - "key_modules": Major modules or components identified
  - "architecture_type": Architecture type suggested by structure
  - "notable_patterns": Any notable organizational patterns
- "structure_insights": Hierarchical representation of important structures
- "key_files": Files that appear central to the architecture

IMPORTANT: Focus on architectural patterns rather than listing every file and directory.
"""

signature_analysis_prompt="""You are an expert code analyst. Your task is to analyze function and class signatures to understand interfaces and contracts.

<ORIGINAL_QUERY>
{original_query}
</ORIGINAL_QUERY>

<CURRENT_QUERY>
{current_query}
</CURRENT_QUERY>

<SIGNATURE_RESULTS>
{signature_results}
</SIGNATURE_RESULTS>

Review the signature results, which show function and class declarations. Extract key insights about:
1. API design patterns and interfaces
2. Parameter and return type patterns
3. Class hierarchies and inheritance
4. Code contracts and interface boundaries

Format your response as a JSON object with the following keys:
- "insights": An object containing structured insights from signature analysis
  - "api_patterns": Common API design patterns identified
  - "parameter_patterns": Patterns in parameter usage
  - "type_system": Insights about the type system usage
  - "class_relationships": Class hierarchy and inheritance patterns
- "signature_insights": Structured representation of key signatures
- "interface_boundaries": Clear interface boundaries identified

IMPORTANT: Focus on understanding interfaces and contracts rather than implementation details.
"""

content_analysis_prompt="""You are an expert code analyst. Your task is to analyze specific content patterns in the code to understand implementation details.

<ORIGINAL_QUERY>
{original_query}
</ORIGINAL_QUERY>

<CURRENT_QUERY>
{current_query}
</CURRENT_QUERY>

<KNOWLEDGE_GATHERED>
{knowledge_gathered}
</KNOWLEDGE_GATHERED>

First, identify key patterns (strings, regex patterns, code constructs) that would be valuable to search for in the codebase based on the current query and knowledge.

Then, for any content analysis results provided, extract insights about:
1. Implementation patterns and idioms
2. Error handling approaches
3. Configuration and environmental handling
4. Specific feature implementations

Format your response as a JSON object with the following keys:
- "patterns": Array of patterns to search for (when asked to identify patterns)
- "insights": An object containing structured insights from content analysis
  - "implementation_patterns": Common implementation patterns
  - "error_handling": Approaches to error handling
  - "configuration_management": How configuration is managed
  - "feature_implementations": Insights about specific features
- "relevant_files": Array of file paths that contain the patterns

IMPORTANT: Focus on extracting actionable implementation insights related to the original query.
"""

result_integration_prompt="""You are an expert code analyst. Your task is to integrate results from multiple analysis methods into a cohesive understanding.

<ORIGINAL_QUERY>
{original_query}
</ORIGINAL_QUERY>

<CURRENT_QUERY>
{current_query}
</CURRENT_QUERY>

<ANALYSIS_RESULTS>
{analysis_results}
</ANALYSIS_RESULTS>

<RELEVANT_FILES>
{relevant_files}
</RELEVANT_FILES>

Review the results from different analysis methods and integrate them into a cohesive understanding of the codebase. Consider:
1. How different insights complement each other
2. Contradictions between different analysis methods
3. A holistic view of the codebase architecture
4. How the integrated knowledge answers the original query

Format your response as a JSON object with the following keys:
- "integrated_knowledge": A structured representation of key knowledge
  - "architecture": Architectural understanding
  - "components": Key component understanding
  - "patterns": Important patterns discovered
  - "workflows": Relevant workflow understanding
- "key_findings": Array of the most important discoveries
- "knowledge_confidence": Assessment of confidence in different areas
- "query_relevance": How the integrated knowledge addresses the original query

IMPORTANT: Focus on creating a cohesive understanding rather than merely summarizing each analysis method.
"""

knowledge_gap_prompt="""You are an expert code analyst. Your task is to identify knowledge gaps and determine the next analysis steps.

<ORIGINAL_QUERY>
{original_query}
</ORIGINAL_QUERY>

<CURRENT_QUERY>
{current_query}
</CURRENT_QUERY>

<KNOWLEDGE_GATHERED>
{knowledge_gathered}
</KNOWLEDGE_GATHERED>

<ANALYSIS_HISTORY>
{analysis_history}
</ANALYSIS_HISTORY>

Review the current knowledge and analysis history to identify significant gaps in understanding the codebase. Consider:
1. Aspects of the original query that remain unanswered
2. Areas where knowledge is incomplete or uncertain
3. Components or relationships that need deeper investigation
4. The most suitable next analysis method to fill these gaps

Format your response as a JSON object with the following keys:
- "knowledge_gaps": Array of specific knowledge gaps identified
- "next_approach": The most appropriate next analysis method to use:
  - "semantic_search"
  - "reference_analysis" 
  - "dependency_analysis"
  - "file_structure_analysis"
  - "signature_analysis"
  - "content_analysis"
- "refined_query": A refined version of the query to target the gaps
- "rationale": Brief explanation of your recommendation

IMPORTANT: Focus on identifying the most critical knowledge gaps and selecting the most efficient approach to fill them.
"""

finalization_prompt="""You are an expert code analyst. Your task is to create a comprehensive final summary of your code analysis.

<ORIGINAL_QUERY>
{original_query}
</ORIGINAL_QUERY>

<KNOWLEDGE_GATHERED>
{knowledge_gathered}
</KNOWLEDGE_GATHERED>

<RELEVANT_FILES>
{relevant_files}
</RELEVANT_FILES>

<ANALYSIS_HISTORY>
{analysis_history}
</ANALYSIS_HISTORY>

Create a comprehensive, well-structured summary that:
1. Directly answers the original query with depth and precision
2. Explains the codebase architecture and key components
3. Details important relationships and patterns discovered
4. Presents the most relevant files and code sections
5. Organizes information in a clear, logical structure with appropriate headings

Your response should read like an expert analysis document. Include:
- A concise executive summary at the top
- Clearly labeled sections for different aspects of the analysis
- Code examples where helpful (properly formatted)
- Architecture diagrams described in text when relevant
- References to specific files and components
- Remaining uncertainties or areas for further investigation

IMPORTANT: This is your final output to the user, so make it thorough, precise, and actionable. Format the response for maximum readability.
"""