import operator
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from typing_extensions import TypedDict, Annotated

@dataclass(kw_only=True)
class CodeAnalysisState:
    # Core query information
    original_query: str = field(default=None)  # Original research query from user
    current_query: str = field(default=None)   # Current search query (may differ from original)
    project_path: str = field(default=None)    # Path to the project being analyzed
    
    # Analysis state tracking
    analysis_results: Dict[str, Any] = field(default_factory=dict)  # Results from different analysis methods
    relevant_files: Set[str] = field(default_factory=set)           # Set of discovered relevant files
    current_file: str = field(default=None)                         # Current file being analyzed
    
    # Knowledge management
    knowledge_gathered: Dict[str, Any] = field(default_factory=dict)  # Structured knowledge about the codebase
    knowledge_gaps: List[str] = field(default_factory=list)           # Identified knowledge gaps
    
    # Analysis history and approach
    analysis_history: Annotated[List[Dict[str, Any]], operator.add] = field(default_factory=list)  # History of analysis steps
    current_approach: str = field(default=None)  # Current analysis approach being used
    analysis_loop_count: int = field(default=0)  # Number of analysis iterations completed
    
    # Task information
    task_category: str = field(default=None)     # Categorization of the task type
    task_complexity: str = field(default=None)   # Assessment of task complexity
    
    # Final result
    analysis_summary: str = field(default=None)  # Final analysis summary

@dataclass(kw_only=True)
class CodeAnalysisStateInput:
    original_query: str = field(default=None)    # Original research query from user
    project_path: str = field(default=None)      # Path to the project being analyzed

@dataclass(kw_only=True)
class CodeAnalysisStateOutput:
    analysis_summary: str = field(default=None)  # Final analysis summary
    relevant_files: Set[str] = field(default_factory=set)  # Set of discovered relevant files
    knowledge_gathered: Dict[str, Any] = field(default_factory=dict)  # Structured knowledge about the codebase