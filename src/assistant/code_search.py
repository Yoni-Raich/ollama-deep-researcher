import os
import re
import glob
import subprocess
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path

from assistant.configuration import Configuration
from assistant.llm_wrapper import LLMWrapper, RetryAzureOpenAIEmbeddings
from assistant.SemanticCodeSearch import SemanticCodeSearch


class CodeSearchWrapper:
    """
    A wrapper for code search functionality that provides a unified interface for
    various code analysis methods.
    """
    
    def __init__(self, config: Configuration, llm_wrapper: LLMWrapper):
        """
        Initialize the CodeSearchWrapper with configuration and LLM wrapper.
        
        Args:
            config: The Configuration object containing settings
            llm_wrapper: The LLMWrapper for LLM interactions
        """
        self.config = config
        self.llm_wrapper = llm_wrapper
        
        # Initialize semantic search if enabled
        self.semantic_search = None
        if config.enable_semantic_search:
            try:
                # Check if the vector database exists
                vector_db_exists = os.path.exists(config.vector_db_path)
                
                # Create semantic code search instance
                if config.llm_provider.value == "azure_openai" and config.azure_openai_embedding_deployment:
                    self.semantic_search = SemanticCodeSearch(
                        project_path=config.project_path,
                        vector_db_path=config.vector_db_path,
                        azure_endpoint=config.azure_openai_endpoint,
                        azure_deployment=config.azure_openai_embedding_deployment,
                        azure_api_version=config.azure_openai_api_version
                    )
                    
                    # Load or build the vector database
                    if vector_db_exists:
                        self.semantic_search.load_vector_db()
                    else:
                        self.semantic_search.build_vector_db()
                        self.semantic_search.save_vector_db()
            except Exception as e:
                print(f"Error initializing semantic search: {str(e)}")
                self.semantic_search = None
        
        # Initialize tool availability flags
        self._initialize_tools()
    
    def _initialize_tools(self) -> None:
        """Initialize and check availability of external tools."""
        # Check for tree-sitter availability if enabled
        self.tree_sitter_available = False
        if self.config.use_tree_sitter:
            try:
                # Try to import tree-sitter
                import tree_sitter
                self.tree_sitter_available = True
            except ImportError:
                print("tree-sitter not available. Falling back to regex-based parsing.")
        
        # Check for ctags availability if enabled
        self.ctags_available = False
        if self.config.use_ctags:
            try:
                # Try to run ctags --version
                result = subprocess.run(
                    ["ctags", "--version"], 
                    capture_output=True, 
                    text=True,
                    check=False
                )
                self.ctags_available = result.returncode == 0
            except Exception:
                print("ctags not available. Falling back to grep-based reference analysis.")
    
    def semantic_code_search(self, query: str, k: int = 5, exclude_files: Optional[List[str]] = None) -> List[str]:
        """
        Perform semantic code search using the SemanticCodeSearch implementation.
        
        Args:
            query: The query to search for
            k: The number of results to return
            exclude_files: A list of files to exclude from the search
            
        Returns:
            A list of file paths that are semantically relevant to the query
        """
        if self.semantic_search is None:
            return []
        
        return self.semantic_search.semantic_search(query, k, exclude_files)
    
    def reference_analysis(self, entity: str, file_types: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find references to a specific entity (function, class, variable) in the codebase.
        
        Args:
            entity: The entity to search for references to
            file_types: Optional list of file extensions to search in
            
        Returns:
            A dictionary mapping file paths to lists of references
        """
        results = {}
        
        # Use ctags if available
        if self.ctags_available:
            return self._ctags_reference_analysis(entity, file_types)
        
        # Fallback to grep-based reference analysis
        return self._grep_reference_analysis(entity, file_types)
    
    def _ctags_reference_analysis(self, entity: str, file_types: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Use ctags for reference analysis."""
        results = {}
        
        try:
            # Create a tags file
            tag_file = os.path.join(self.config.project_path, "tags")
            file_pattern = ""
            if file_types:
                file_pattern = f" --languages={','.join(file_types)}"
            
            subprocess.run(
                f"ctags -R --fields=+n{file_pattern} .",
                shell=True,
                cwd=self.config.project_path,
                check=True,
                capture_output=True
            )
            
            # Parse the tags file
            with open(tag_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('!'):  # Skip header lines
                        continue
                    
                    parts = line.strip().split('\t')
                    if len(parts) >= 3 and parts[0] == entity:
                        file_path = parts[1]
                        if file_path not in results:
                            results[file_path] = []
                        
                        # Extract line number if available
                        line_num = None
                        pattern = parts[2]
                        line_match = re.search(r'line:(\d+)', '\t'.join(parts[3:]))
                        if line_match:
                            line_num = int(line_match.group(1))
                        
                        results[file_path].append({
                            'entity': entity,
                            'line': line_num,
                            'pattern': pattern
                        })
                        
            # Clean up tags file
            if os.path.exists(tag_file):
                os.remove(tag_file)
                
        except Exception as e:
            print(f"Error performing ctags reference analysis: {str(e)}")
            # Fall back to grep if ctags fails
            return self._grep_reference_analysis(entity, file_types)
            
        return results
    
    def _grep_reference_analysis(self, entity: str, file_types: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Use grep for reference analysis."""
        results = {}
        
        try:
            # Build file pattern
            file_pattern = ""
            if file_types:
                file_pattern = f" --include='*.{{{','.join(file_types)}}}'"
            
            # Run grep
            grep_cmd = f"grep -r{file_pattern} -n '\\b{entity}\\b' {self.config.project_path}"
            output = subprocess.run(
                grep_cmd,
                shell=True,
                capture_output=True,
                text=True,
                check=False
            )
            
            # Parse output
            if output.returncode <= 1:  # 0 = matches found, 1 = no matches
                for line in output.stdout.splitlines():
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        file_path = parts[0]
                        line_num = int(parts[1])
                        context = parts[2].strip()
                        
                        if file_path not in results:
                            results[file_path] = []
                        
                        results[file_path].append({
                            'entity': entity,
                            'line': line_num,
                            'context': context
                        })
                        
        except Exception as e:
            print(f"Error performing grep reference analysis: {str(e)}")
            
        return results
    
    def dependency_analysis(self, file_path: str) -> Dict[str, List[str]]:
        """
        Analyze dependencies for a specific file.
        
        Args:
            file_path: The path to the file to analyze
            
        Returns:
            A dictionary with 'imports' and 'imported_by' lists
        """
        result = {
            'imports': [],
            'imported_by': []
        }
        
        try:
            # Get file extension
            ext = os.path.splitext(file_path)[1].lower()
            
            # Read the file content
            with open(os.path.join(self.config.project_path, file_path), 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract imports based on file type
            if ext in ['.py']:
                # Python imports
                import_patterns = [
                    r'^\s*import\s+([^;\n]+)',
                    r'^\s*from\s+([^\s;]+)\s+import'
                ]
                for pattern in import_patterns:
                    for match in re.finditer(pattern, content, re.MULTILINE):
                        imports = match.group(1).strip().split(',')
                        for imp in imports:
                            clean_import = imp.strip().split(' as ')[0]
                            if clean_import and clean_import not in result['imports']:
                                result['imports'].append(clean_import)
            
            elif ext in ['.js', '.ts', '.jsx', '.tsx']:
                # JavaScript/TypeScript imports
                import_patterns = [
                    r'^\s*import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]',
                    r'^\s*require\s*\(\s*[\'"]([^\'"]+)[\'"]'
                ]
                for pattern in import_patterns:
                    for match in re.finditer(pattern, content, re.MULTILINE):
                        imp = match.group(1).strip()
                        if imp and imp not in result['imports']:
                            result['imports'].append(imp)
            
            elif ext in ['.java']:
                # Java imports
                import_pattern = r'^\s*import\s+([^;]+);'
                for match in re.finditer(import_pattern, content, re.MULTILINE):
                    imp = match.group(1).strip()
                    if imp and imp not in result['imports']:
                        result['imports'].append(imp)
            
            elif ext in ['.cs']:
                # C# imports
                import_pattern = r'^\s*using\s+([^;]+);'
                for match in re.finditer(import_pattern, content, re.MULTILINE):
                    imp = match.group(1).strip()
                    if imp and imp not in result['imports']:
                        result['imports'].append(imp)
            
            # Find files that import this file
            file_name = os.path.basename(file_path)
            name_without_ext = os.path.splitext(file_name)[0]
            
            # Search for files that might import this file
            grep_cmd = f"grep -r -l '\\b{name_without_ext}\\b' {self.config.project_path}"
            output = subprocess.run(
                grep_cmd,
                shell=True,
                capture_output=True,
                text=True,
                check=False
            )
            
            if output.returncode <= 1:  # 0 = matches found, 1 = no matches
                for importing_file in output.stdout.splitlines():
                    # Skip the file itself
                    if os.path.abspath(importing_file) == os.path.abspath(os.path.join(self.config.project_path, file_path)):
                        continue
                    
                    # Get relative path
                    rel_path = os.path.relpath(importing_file, self.config.project_path)
                    if rel_path not in result['imported_by']:
                        result['imported_by'].append(rel_path)
            
        except Exception as e:
            print(f"Error analyzing dependencies for {file_path}: {str(e)}")
            
        return result
    
    def file_structure_analysis(self, directory: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze the file structure of the project or a specific directory.
        
        Args:
            directory: Optional directory to analyze (relative to project path).
                      If None, analyzes the entire project.
            
        Returns:
            A dictionary containing the file structure
        """
        result = {
            'directories': [],
            'files': [],
            'stats': {
                'total_dirs': 0,
                'total_files': 0,
                'file_types': {}
            }
        }
        
        try:
            # Determine the root directory to analyze
            root_dir = self.config.project_path
            if directory:
                root_dir = os.path.join(root_dir, directory)
            
            # Walk the directory
            for dirpath, dirnames, filenames in os.walk(root_dir):
                # Skip hidden directories
                dirnames[:] = [d for d in dirnames if not d.startswith('.')]
                
                # Calculate relative path
                rel_path = os.path.relpath(dirpath, self.config.project_path)
                if rel_path != '.':
                    result['directories'].append(rel_path)
                
                # Process files
                for filename in filenames:
                    # Skip hidden files
                    if filename.startswith('.'):
                        continue
                        
                    # Get file extension
                    _, ext = os.path.splitext(filename)
                    ext = ext.lower()
                    
                    # Update stats
                    if ext not in result['stats']['file_types']:
                        result['stats']['file_types'][ext] = 0
                    result['stats']['file_types'][ext] += 1
                    
                    # Add file to result
                    file_path = os.path.join(rel_path, filename)
                    if file_path != '.':
                        result['files'].append(file_path)
            
            # Update total counts
            result['stats']['total_dirs'] = len(result['directories'])
            result['stats']['total_files'] = len(result['files'])
            
        except Exception as e:
            print(f"Error analyzing file structure: {str(e)}")
            
        return result
    
    def signature_analysis(self, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze function/method signatures and class structures in a file.
        
        Args:
            file_path: The path to the file to analyze
            
        Returns:
            A dictionary with 'classes' and 'functions' lists containing signature information
        """
        result = {
            'classes': [],
            'functions': []
        }
        
        try:
            # Get file extension
            ext = os.path.splitext(file_path)[1].lower()
            
            # Read the file content
            with open(os.path.join(self.config.project_path, file_path), 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use tree-sitter if available
            if self.tree_sitter_available:
                return self._tree_sitter_signature_analysis(file_path, content, ext)
            
            # Fallback to regex-based analysis
            return self._regex_signature_analysis(file_path, content, ext)
            
        except Exception as e:
            print(f"Error analyzing signatures for {file_path}: {str(e)}")
            
        return result
    
    def _tree_sitter_signature_analysis(self, file_path: str, content: str, ext: str) -> Dict[str, List[Dict[str, Any]]]:
        """Use tree-sitter for signature analysis."""
        # This would be implemented with tree-sitter for more accurate parsing
        # For now, fall back to regex-based parsing
        return self._regex_signature_analysis(file_path, content, ext)
    
    def _regex_signature_analysis(self, file_path: str, content: str, ext: str) -> Dict[str, List[Dict[str, Any]]]:
        """Use regex for signature analysis."""
        result = {
            'classes': [],
            'functions': []
        }
        
        # Python
        if ext == '.py':
            # Class pattern
            class_pattern = r'^\s*class\s+(\w+)(?:\s*\(\s*([^)]*)\s*\))?:'
            for match in re.finditer(class_pattern, content, re.MULTILINE):
                class_name = match.group(1)
                class_parents = [p.strip() for p in match.group(2).split(',')] if match.group(2) else []
                
                result['classes'].append({
                    'name': class_name,
                    'extends': class_parents,
                    'line': content[:match.start()].count('\n') + 1
                })
            
            # Function pattern
            func_pattern = r'^\s*def\s+(\w+)\s*\(\s*([^)]*)\s*\)(?:\s*->\s*([^:]+))?:'
            for match in re.finditer(func_pattern, content, re.MULTILINE):
                func_name = match.group(1)
                params = [p.strip() for p in match.group(2).split(',')] if match.group(2) else []
                return_type = match.group(3).strip() if match.group(3) else None
                
                result['functions'].append({
                    'name': func_name,
                    'params': params,
                    'return_type': return_type,
                    'line': content[:match.start()].count('\n') + 1
                })
                
        # JavaScript/TypeScript
        elif ext in ['.js', '.ts', '.jsx', '.tsx']:
            # Class pattern
            class_pattern = r'^\s*(?:export\s+)?(?:default\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?'
            for match in re.finditer(class_pattern, content, re.MULTILINE):
                class_name = match.group(1)
                parent_class = match.group(2) if match.group(2) else None
                
                result['classes'].append({
                    'name': class_name,
                    'extends': [parent_class] if parent_class else [],
                    'line': content[:match.start()].count('\n') + 1
                })
            
            # Function pattern (various forms)
            func_patterns = [
                r'^\s*(?:export\s+)?(?:default\s+)?function\s+(\w+)\s*\(\s*([^)]*)\s*\)',
                r'^\s*(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[^=]*)\s*=>\s*[{(]',
                r'^\s*(?:async\s+)?(\w+)\s*\(\s*([^)]*)\s*\)\s*[{]'
            ]
            
            for pattern in func_patterns:
                for match in re.finditer(pattern, content, re.MULTILINE):
                    func_name = match.group(1)
                    params = [p.strip() for p in match.group(2).split(',')] if len(match.groups()) > 1 and match.group(2) else []
                    
                    result['functions'].append({
                        'name': func_name,
                        'params': params,
                        'line': content[:match.start()].count('\n') + 1
                    })
                    
        # C#
        elif ext == '.cs':
            # Class pattern
            class_pattern = r'^\s*(?:public|private|protected|internal)?\s*(?:static\s+)?class\s+(\w+)(?:\s*:\s*([^{]*))?'
            for match in re.finditer(class_pattern, content, re.MULTILINE):
                class_name = match.group(1)
                class_parents = [p.strip() for p in match.group(2).split(',')] if match.group(2) else []
                
                result['classes'].append({
                    'name': class_name,
                    'extends': class_parents,
                    'line': content[:match.start()].count('\n') + 1
                })
            
            # Method pattern
            method_pattern = r'^\s*(?:public|private|protected|internal)?\s*(?:static\s+)?(?:async\s+)?(\w+(?:<[^>]+>)?)\s+(\w+)\s*\(\s*([^)]*)\s*\)'
            for match in re.finditer(method_pattern, content, re.MULTILINE):
                return_type = match.group(1)
                method_name = match.group(2)
                params = [p.strip() for p in match.group(3).split(',')] if match.group(3) else []
                
                result['functions'].append({
                    'name': method_name,
                    'params': params,
                    'return_type': return_type,
                    'line': content[:match.start()].count('\n') + 1
                })
                
        # Java
        elif ext == '.java':
            # Class pattern
            class_pattern = r'^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([^{]*))?'
            for match in re.finditer(class_pattern, content, re.MULTILINE):
                class_name = match.group(1)
                parent_class = match.group(2) if match.group(2) else None
                interfaces = [i.strip() for i in match.group(3).split(',')] if match.group(3) else []
                
                result['classes'].append({
                    'name': class_name,
                    'extends': [parent_class] if parent_class else [],
                    'implements': interfaces,
                    'line': content[:match.start()].count('\n') + 1
                })
            
            # Method pattern
            method_pattern = r'^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?(\w+(?:<[^>]+>)?)\s+(\w+)\s*\(\s*([^)]*)\s*\)'
            for match in re.finditer(method_pattern, content, re.MULTILINE):
                return_type = match.group(1)
                method_name = match.group(2)
                params = [p.strip() for p in match.group(3).split(',')] if match.group(3) else []
                
                result['functions'].append({
                    'name': method_name,
                    'params': params,
                    'return_type': return_type,
                    'line': content[:match.start()].count('\n') + 1
                })
                
        return result
    
    def content_analysis(self, query: str, file_types: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for specific strings or patterns in code.
        
        Args:
            query: The pattern to search for
            file_types: Optional list of file extensions to search in
            
        Returns:
            A dictionary mapping file paths to lists of matches
        """
        results = {}
        
        try:
            # Build file pattern
            file_pattern = ""
            if file_types:
                file_pattern = f" --include='*.{{{','.join(file_types)}}}'"
            
            # Run grep
            grep_cmd = f"grep -r{file_pattern} -n '{query}' {self.config.project_path}"
            output = subprocess.run(
                grep_cmd,
                shell=True,
                capture_output=True,
                text=True,
                check=False
            )
            
            # Parse output
            if output.returncode <= 1:  # 0 = matches found, 1 = no matches
                for line in output.stdout.splitlines():
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        file_path = parts[0]
                        line_num = int(parts[1])
                        context = parts[2].strip()
                        
                        if file_path not in results:
                            results[file_path] = []
                        
                        results[file_path].append({
                            'line': line_num,
                            'context': context
                        })
                        
        except Exception as e:
            print(f"Error performing content analysis: {str(e)}")
            
        return results 