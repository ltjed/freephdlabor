"""Knowledge Base Repo Retrieval Tools
This module contains tools for semantic and keyword search in a knowledge base stored as a structured repository.
The knowledge base is organized into folders and files, allowing for efficient retrieval of information.
"""

from smolagents.tools import Tool
from general_tools.kb_repo_management.repo_indexer import RepoIndexer
import os
import shutil
from pathlib import Path
import time

class SemanticSearchKnowledgeBase(Tool):
    name = "semantic_search_knowledge_base"
    description = (
        "Perform a semantic search in the knowledge base. "
        "Returns the path and content of most relevant files or code snippets for a given query."
    )
    inputs = {"query": {"type": "string", "description": "The search query."}}
    output_type = "string"

    def __init__(self, repo_indexer: RepoIndexer):
        super().__init__()
        self.repo_indexer = repo_indexer

    def forward(self, query: str) -> str:
        # return self.repo_indexer.get_query_results(query, k=3)
        return self.repo_indexer.get_unique_query_results(query, k=3)

class KeywordSearchKnowledgeBase(Tool):
    name = "keyword_search_knowledge_base"
    description = (
        "Search for a keyword in a plain text file or recursively in all plain text files within a folder. "
        "Returns matching lines with file names, line numbers and context lines before and after each match. "
        "Only supports plain text files (e.g., .txt, .py, .md). Not suitable for binary formats like .pdf, .docx, .xlsx."
    )
    inputs = {
        "path": {"type": "string", "description": "Path to the file or folder to search in."},
        "keyword": {"type": "string", "description": "Keyword to search for."},
        "context_lines": {
            "type": "integer",
            "description": "Number of lines to include before and after each match."
        }
    }
    output_type = "string"

    ALLOWED_EXTENSIONS = {".py", ".md", ".txt"}
    MAX_SIZE = 2 * 1024 * 1024  # 2 MB

    def __init__(self, repo_indexer: RepoIndexer, max_search_time: int = 10):
        super().__init__()
        self.knowledge_base_dir = Path(repo_indexer.root)
        self.max_search_time = max_search_time  # seconds

    def forward(self, path: str, keyword: str, context_lines: int) -> str:
        # Disallow absolute paths
        if os.path.isabs(path):
            return "Error: Absolute paths are not allowed. Please specify a relative path within the knowledge base. For example use '.' to search in the root of knowledge base."
        # Prevent directory traversal
        target_path = (self.knowledge_base_dir / path).resolve()
        if not str(target_path).startswith(str(self.knowledge_base_dir.resolve())):
            return "Error: Path escapes the knowledge base directory."

        start_time = time.time()
        if not target_path.exists():
            return f"The path '{path}' does not exist."

        def timed_out():
            return (time.time() - start_time) > self.max_search_time

        if os.path.isfile(target_path):
            if not any(target_path.endswith(ext) for ext in self.ALLOWED_EXTENSIONS):
                return f"[{path}]: Skipped (unsupported file type)"
            if os.path.getsize(target_path) > self.MAX_SIZE:
                return f"[{path}]: Skipped (file too large)"
            return self._search_in_file(target_path, keyword, context_lines, display_path=path)
        elif os.path.isdir(target_path):
            results = []
            for root, _, files in os.walk(target_path):
                for fname in files:
                    if timed_out():
                        return (
                            f"Maximum search time ({self.max_search_time} seconds) reached while searching '{path}'.\n"
                            "The knowledge base may be too large for keyword search. "
                            "Consider using semantic search instead for faster and more relevant results."
                        )
                    if not any(fname.endswith(ext) for ext in self.ALLOWED_EXTENSIONS):
                        continue
                    fpath = os.path.join(root, fname)
                    rel_path = os.path.relpath(fpath, self.knowledge_base_dir)
                    if os.path.getsize(fpath) > self.MAX_SIZE:
                        results.append(f"[{rel_path}]: Skipped (file too large)")
                        continue
                    try:
                        result = self._search_in_file(fpath, keyword, context_lines, display_path=rel_path)
                        if "No matches found" not in result:
                            results.append(result)
                    except Exception as e:
                        results.append(f"[{rel_path}]: Error reading file ({e})")
            return "\n\n".join(results) if results else f"No matches found for '{keyword}' in folder '{path}'."
        else:
            return f"The path '{path}' is neither a file nor a directory."

    def _search_in_file(self, filepath: str, keyword: str, context_lines: int, display_path: str) -> str:
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            return f"[{display_path}]: Cannot read binary or non-text file."

        num_lines = len(lines)
        match_indices = [i for i, line in enumerate(lines) if keyword in line]

        if not match_indices:
            return f"[{display_path}]: No matches found for '{keyword}'."

        output_lines = set()
        for idx in match_indices:
            start = max(0, idx - context_lines)
            end = min(num_lines, idx + context_lines + 1)
            output_lines.update(range(start, end))

        sorted_output = sorted(output_lines)
        formatted_output = [f"{i+1}: {lines[i].rstrip()}" for i in sorted_output]

        return f"--- Matches in [{display_path}] ---\n" + "\n".join(formatted_output)

class CopyFromKnowledgeBase(Tool): 
    name = "copy_from_knowledge_base"
    description = (
        "Copy a file or folder from the knowledge base to the working directory. "
        "If `overwrite=True`, it will replace existing files or merge directories, overwriting same-name files. "
        "If `overwrite=False`, it will avoid conflicts by adding a numeric suffix to the destination name."
    )
    inputs = {
        "source_path": {"type": "string", "description": "Relative path in the knowledge base."},
        "destination_path": {"type": "string", "description": "Relative path in the working directory."},
        "overwrite": {"type": "boolean", "description": "Whether to overwrite existing files or merge folders."}
    }
    output_type = "string"

    def __init__(self, repo_indexer: RepoIndexer, working_dir: str):
        super().__init__()
        self.repo_indexer = repo_indexer
        self.working_dir = Path(working_dir)
        self.root = Path(repo_indexer.root)

    def _get_unique_path(self, base_path: Path) -> Path:
        """Returns a path with a numeric suffix that avoids conflict."""
        counter = 1
        new_path = base_path
        while new_path.exists():
            new_path = base_path.with_name(f"{base_path.stem}_{counter}{base_path.suffix}")
            counter += 1
        return new_path

    def _safe_kb_path(self, path: str) -> Path:
        abs_root = self.root.resolve()
        abs_path = (self.root / path).resolve()
        if not str(abs_path).startswith(str(abs_root)):
            raise PermissionError("Access outside the knowledge base root is not allowed.")
        return abs_path

    def _safe_working_path(self, path: str) -> Path:
        abs_root = self.working_dir.resolve()
        abs_path = (self.working_dir / path).resolve()
        if not str(abs_path).startswith(str(abs_root)):
            raise PermissionError("Access outside the working directory is not allowed.")
        return abs_path

    def forward(self, source_path: str, destination_path: str, overwrite: bool) -> str:
        try:
            src = self._safe_kb_path(source_path)
            dst = self._safe_working_path(destination_path)
        except PermissionError as e:
            return str(e)
        
        if not src.exists():
            return f"Error: source path '{source_path}' does not exist in the knowledge base."

        if dst.exists():
            if overwrite:
                if src.is_dir():
                    dst.mkdir(parents=True, exist_ok=True)
                    for item in src.rglob("*"):
                        target = dst / item.relative_to(src)
                        if item.is_dir():
                            target.mkdir(parents=True, exist_ok=True)
                        else:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(item, target)
                    return f"Directory '{source_path}' merged into existing '{destination_path}' with overwrites."
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    return f"File '{source_path}' copied to existing '{destination_path}', overwriting old file."
            else:
                dst = self._get_unique_path(dst)

        if src.is_dir():
            shutil.copytree(src, dst)
            return f"Directory '{source_path}' copied to '{dst.relative_to(self.working_dir)}' (no overwrite)."
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return f"File '{source_path}' copied to '{dst.relative_to(self.working_dir)}' (no overwrite)."
