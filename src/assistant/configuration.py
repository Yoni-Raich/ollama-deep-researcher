import os
from dataclasses import dataclass, fields, field
from typing import Any, Optional, List

from langchain_core.runnables import RunnableConfig
from dataclasses import dataclass

from enum import Enum

class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"

class LLMProvider(Enum):
    OLLAMA = "ollama"
    AZURE_OPENAI = "azure_openai"

@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the code analysis system."""
    # Core settings
    max_analysis_loops: int = int(os.environ.get("MAX_ANALYSIS_LOOPS", "5"))
    project_path: str = os.environ.get("PROJECT_PATH", ".")
    documents_path: str = os.environ.get("DOCUMENTS_PATH", "./documents")
    vector_db_path: str = os.environ.get("VECTOR_DB_PATH", "./vector_db")
    
    # LLM settings
    llm_provider: LLMProvider = LLMProvider(os.environ.get("LLM_PROVIDER", LLMProvider.OLLAMA.value))
    local_llm: str = os.environ.get("OLLAMA_MODEL", "llama3.2")
    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/")
    
    # Azure OpenAI settings
    azure_openai_endpoint: str = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_api_key: str = os.environ.get("AZURE_OPENAI_API_KEY", "")
    azure_openai_deployment: str = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
    azure_openai_api_version: str = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15")
    azure_openai_embedding_deployment: str = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")
    
    # External web search (for context when needed)
    search_api: SearchAPI = SearchAPI(os.environ.get("SEARCH_API", SearchAPI.DUCKDUCKGO.value))
    fetch_full_page: bool = os.environ.get("FETCH_FULL_PAGE", "False").lower() in ("true", "1", "t")
    
    # Analysis method settings
    enable_semantic_search: bool = os.environ.get("ENABLE_SEMANTIC_SEARCH", "True").lower() in ("true", "1", "t")
    enable_reference_analysis: bool = os.environ.get("ENABLE_REFERENCE_ANALYSIS", "True").lower() in ("true", "1", "t")
    enable_dependency_analysis: bool = os.environ.get("ENABLE_DEPENDENCY_ANALYSIS", "True").lower() in ("true", "1", "t")
    enable_file_structure_analysis: bool = os.environ.get("ENABLE_FILE_STRUCTURE_ANALYSIS", "True").lower() in ("true", "1", "t")
    enable_signature_analysis: bool = os.environ.get("ENABLE_SIGNATURE_ANALYSIS", "True").lower() in ("true", "1", "t")
    enable_content_analysis: bool = os.environ.get("ENABLE_CONTENT_ANALYSIS", "True").lower() in ("true", "1", "t")
    
    # Tool-specific settings
    use_tree_sitter: bool = os.environ.get("USE_TREE_SITTER", "False").lower() in ("true", "1", "t")
    use_ctags: bool = os.environ.get("USE_CTAGS", "False").lower() in ("true", "1", "t")
    
    # Search depth limits
    max_file_size: int = int(os.environ.get("MAX_FILE_SIZE", "1000000"))  # 1MB default
    max_files_to_analyze: int = int(os.environ.get("MAX_FILES_TO_ANALYZE", "50"))
    max_depth: int = int(os.environ.get("MAX_DEPTH", "5"))
    max_dependencies: int = int(os.environ.get("MAX_DEPENDENCIES", "10"))
    
    # Language/framework specific settings
    supported_languages: List[str] = field(
        default_factory=lambda: os.environ.get("SUPPORTED_LANGUAGES", "python,javascript,typescript,java,csharp,go,rust").split(",")
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})