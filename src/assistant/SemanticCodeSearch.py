import math
import os
import glob
import re
import time
from typing import List, Optional
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

# Define a helper to detect rate limit errors
def is_rate_limit_error(exception):
    return isinstance(exception, httpx.HTTPStatusError) and exception.response.status_code == 429

# Create a subclass that adds retry logic to embedding calls
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

class SemanticCodeSearch:
    def __init__(self, project_path: str, vector_db_path: str,
                 azure_endpoint: str, azure_deployment: str, azure_api_version: str,
                 chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the SemanticCodeSearch instance.

        Parameters:
        - project_path: The root directory of the C# project.
        - vector_db_path: The local path to save/load the vector database.
        - azure_endpoint: Azure OpenAI endpoint URL.
        - azure_deployment: Name of the Azure OpenAI deployment.
        - azure_api_version: The API version to use with Azure OpenAI.
        - chunk_size: Maximum size of text chunks for splitting documents.
        - chunk_overlap: Overlap size between chunks.
        """
        self.project_path = project_path
        self.vector_db_path = vector_db_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize the Azure OpenAI embeddings model with retry logic.
        self.embeddings = RetryAzureOpenAIEmbeddings(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            openai_api_version=azure_api_version,
        )
        self.vector_db = None

    def load_vector_db(self):
        """
        Load the vector database from local storage.
        WARNING: Enabling dangerous deserialization can execute arbitrary code if the file is untrusted.
        Make sure the file comes from a trusted source.
        """
        if os.path.exists(self.vector_db_path):
            self.vector_db = FAISS.load_local(
                self.vector_db_path,
                self.embeddings,
                allow_dangerous_deserialization=True  # Enable only if the file is trusted!
            )
            print(f"Loaded vector DB from {self.vector_db_path}")
        else:
            print("Vector DB file does not exist at the specified path.")


    def save_vector_db(self):
        """
        Save the current vector database to local storage.
        """
        if self.vector_db is not None:
            self.vector_db.save_local(self.vector_db_path)
            print(f"Vector DB saved to {self.vector_db_path}")
        else:
            print("No vector DB available to save.")

    def build_vector_db(self):
        """
        Builds a vector database for C# projects with enhanced code structure awareness.
        Features:
        - Code-aware text splitting with C# specific separators
        - Metadata enrichment (namespace, class, method)
        - Syntax validation for C# files
        - Batch processing with progress tracking
        """
        documents = []
        
        # Recursively find all .cs files with parallel processing
        cs_files = []
        for root, _, files in os.walk(self.project_path):
            cs_files.extend(os.path.join(root, f) for f in files if f.endswith('.cs'))
        
        if not cs_files:
            print("No C# files found in the project.")
            return

        # Process files with basic syntax validation
        for file_path in cs_files:
            try:
                # Basic C# syntax validation
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not re.search(r'\b(class|namespace|using)\b', content):
                        print(f"Skipping non-C# file: {file_path}")
                        continue

                loader = TextLoader(file_path)
                docs = loader.load()
                
                # Extract file-level metadata
                file_metadata = self._extract_csharp_metadata(content)
                
                for doc in docs:
                    doc.metadata = {
                        "source": file_path,
                        **file_metadata,
                        "file_type": "C#",
                        "lines": len(content.split('\n'))
                    }
                documents.extend(docs)
            except UnicodeDecodeError:
                print(f"Encoding error in {file_path}, try UTF-8 or other encoding.")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

        if not documents:
            print("No valid C# documents found.")
            return

        # Code-aware text splitting with C# specific parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                '\n\nnamespace ',  # Namespace declarations
                '\n\nclass ',      # Class declarations
                '\n\ninterface ', # Interface declarations
                '\n\npublic ',    # Public members
                '\n\nprivate ',    # Private members
                '\n}',            # End of blocks
                '\n//',           # Comments
                '\n\n',            # Double newlines
                '\n',             # Single newlines
                ' ',              # Spaces
                ''                 # Everything else
            ]
        )
        
        splitted_docs = text_splitter.split_documents(documents)
        
        # Enrich chunks with local context metadata
        for doc in splitted_docs:
            chunk_metadata = self._extract_csharp_metadata(doc.page_content)
            doc.metadata.update({
                **chunk_metadata,
                "chunk_lines": len(doc.page_content.split('\n')),
                "signature": self._extract_method_signature(doc.page_content)
            })

        total_chunks = len(splitted_docs)
        print(f"Processing {total_chunks} code chunks across {len(documents)} files")

        # Batch processing with dynamic sizing
        batch_size = min(100, max(20, int(total_chunks/50)))  # Dynamic batch sizing
        for i in range(0, total_chunks, batch_size):
            batch_docs = splitted_docs[i:i + batch_size]
            start_time = time.time()
            
            if i == 0:
                self.vector_db = FAISS.from_documents(batch_docs, self.embeddings)
            else:
                self.vector_db.add_documents(batch_docs)
                
            # Adaptive throttling based on processing time
            processing_time = time.time() - start_time
            if processing_time < 1.0:
                time.sleep(1.0 - processing_time)
                
            print(f"Processed batch {i//batch_size + 1}/{(total_chunks//batch_size)+1} "
                f"({min(i+batch_size, total_chunks)/total_chunks:.1%})")

        print(f"Vector DB created with {self.vector_db.index.ntotal} entries.")

    def _extract_csharp_metadata(self, code: str) -> dict:
        """Extracts C# specific metadata from code snippets"""
        metadata = {}
        
        # Namespace detection
        namespace_match = re.search(
            r'namespace\s+([\w\.]+(?:\s*{\s*[\w\W]*?})?)', 
            code,
            re.MULTILINE
        )
        if namespace_match:
            metadata['namespace'] = namespace_match.group(1).strip('{').strip()
        
        # Class/Interface detection
        class_match = re.search(
            r'(class|interface|struct)\s+(\w+)[<\s]', 
            code,
            re.MULTILINE
        )
        if class_match:
            metadata[class_match.group(1)] = class_match.group(2)
        
        # Method detection
        method_match = re.search(
            r'(?:public|private|protected|internal)\s+(?:static\s+)?(\w+(?:<.*>)?)\s+(\w+)\s*\(',
            code,
            re.MULTILINE
        )
        if method_match:
            metadata['method'] = f"{method_match.group(1)} {method_match.group(2)}"
        
        # Using directives count
        metadata['usings'] = len(re.findall(r'^using\s+[^\s;]+;', code, re.MULTILINE))
        
        return metadata

    def _extract_method_signature(self, code: str) -> str:
        """Extracts method signatures from code chunks"""
        signature_match = re.search(
            r'(?:public|private|protected|internal)\s+'
            r'(?:static\s+)?(?:async\s+)?(?:[\w<>]+\s+)+\b(\w+)\s*\([^\)]*\)',
            code
        )
        return signature_match.group(0) if signature_match else "Unknown signature"

    def update_vector_db(self):
        """
        Update the existing vector database with new C# files that have not been indexed yet.
        """
        if self.vector_db is None:
            print("No existing vector DB. Please build one first.")
            return

        # Retrieve file paths already indexed in the vector DB using the _dict attribute of the docstore
        indexed_files = {
            doc.metadata.get("source")
            for doc in self.vector_db.docstore._dict.values()
            if doc.metadata.get("source")
        }
        new_documents = []
        
        # Scan the project for new .cs files.
        for root, dirs, files in os.walk(self.project_path):
            for file in files:
                if file.endswith('.cs'):
                    file_path = os.path.join(root, file)
                    if file_path in indexed_files:
                        continue  # Skip files that are already indexed.
                    try:
                        loader = TextLoader(file_path)
                        docs = loader.load()
                        for doc in docs:
                            if not doc.metadata:
                                doc.metadata = {}
                            doc.metadata["source"] = file_path
                        new_documents.extend(docs)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

        if new_documents:
            text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            splitted_new_docs = text_splitter.split_documents(new_documents)
            self.vector_db.add_documents(splitted_new_docs)
            print(f"Added {len(splitted_new_docs)} new document chunks to the vector DB.")
        else:
            print("No new documents found to update.")


    def semantic_search(self, query: str, k: int = 5, exclude_files: Optional[List[str]] = None):
        """
        Perform a semantic search on the vector database using the provided query,
        while ignoring specific files.

        Parameters:
        - query: The semantic search query.
        - k: The number of top results to return.
        - exclude_files: A list of file paths to exclude from the search results.

        Returns:
        A list of unique file paths that are most semantically similar to the query,
        excluding the specified files.
        """
        if self.vector_db is None:
            print("Vector DB is not loaded or built.")
            return []

        # Define the filter to exclude specific files
        filter = None
        if exclude_files:
            filter = {"source": {"$nin": exclude_files}}

        # Perform the search with the filter
        results = self.vector_db.similarity_search(query, k=k, filter=filter)

        # Extract unique file paths from the results
        file_paths = list({doc.metadata.get("source", "unknown") for doc in results})
        return file_paths

# Example usage:
if __name__ == "__main__":
    # Configure the paths and Azure OpenAI parameters.
    project_path = r"C:\Repos\Work\old_Automation\MfgTools\src"       # Replace with the path to your C# project.
    vector_db_path = "vector_db_old_auto"       # Replace with the desired path for the vector DB.
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
    azure_api_version = os.environ.get("AZ_OPENAI_API_VERSION")
    
    searcher = SemanticCodeSearch(
        project_path=project_path,
        vector_db_path=vector_db_path,
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        azure_api_version=azure_api_version
    )
    
    # Build the vector DB from scratch.
    searcher.build_vector_db()
    #searcher.load_vector_db()
    #searcher.update_vector_db()
    searcher.save_vector_db()
    # Optionally, load an existing vector DB and update it with new files.
    # searcher.load_vector_db()
    # searcher.update_vector_db()
    
    # Execute a semantic search query.
    query = "How to create an MeinfoTest"
    similar_files = searcher.semantic_search(query, k=5)
                                             
    print("Files related to the query:")
    for file in similar_files:
        print(file)
